"""Part C — privacy-first, reproducible support-ticket triage.

Run from the project root: python -m src.triage
"""
from pathlib import Path
from collections import Counter
import json
import random
import re
import time

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
CATEGORIES = {'billing', 'delivery', 'technical', 'fraud'}
PATTERNS = [
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[EMAIL]'),
    ('phone', re.compile(r'\b[6-9]\d{9}\b'), '[PHONE]'),
    ('order_id', re.compile(r'ORD-\d{6}'), '[ORDER_ID]'),
]
ZERO = '''CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud.
Return JSON only in the form {"category": "...", "reason": "..."}.'''
FEW = ZERO + '''
EXAMPLES:
Text: Unauthorized transaction, I suspect fraud. -> {"category":"fraud","reason":"unauthorized transaction"}
Text: My order arrived late. -> {"category":"delivery","reason":"late delivery"}
Text: App crashes at login. -> {"category":"technical","reason":"app crash"}
'''


def redact(text):
    """Replace PII in the required email → phone → order-id order."""
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, count = pattern.subn(placeholder, text)
        counts[name] = count
    return text, counts


def assert_redacted(text):
    assert all(not pattern.search(text) for _, pattern, _ in PATTERNS), 'Unredacted PII detected'


def audit_privacy(tickets):
    """Audit ALL tickets, including blocked tickets; does not call the model."""
    totals = Counter({name: 0 for name, _, _ in PATTERNS})
    for text in tickets['text'].fillna('').astype(str):
        safe, counts = redact(text)
        totals.update(counts)
        assert_redacted(safe)
    totals['total'] = sum(totals.values())
    return dict(totals)


def retry_chat(messages, ticket_id, retry_stats=None):
    """Initial attempt + at most three retries."""
    for retry_no in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no == 3:
                raise
            wait = BASE_DELAY * 2**retry_no * random.uniform(0.9, 1.15)
            print(f'retry ticket={ticket_id} retry={retry_no} wait={wait:.4f}s')
            if retry_stats is not None:
                retry_stats.append((str(ticket_id), retry_no, wait))
            time.sleep(wait)


def validate_answer(content):
    """Valid JSON with an allowed category, or exactly one category word in non-JSON."""
    content = content or ''
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        found = [word for word in re.findall(r'[a-z]+', content.lower()) if word in CATEGORIES]
        return (found[0], 'classified_keyword') if len(found) == 1 else ('', 'flagged_invalid_output')
    if isinstance(obj, dict) and obj.get('category') in CATEGORIES:
        return obj['category'], 'classified_json'
    return '', 'flagged_invalid_output'


def _segments():
    path = OUT / 'customer_segments.csv'
    if not path.exists():
        raise FileNotFoundError(f'Part B segment file required for routing: {path}')
    seg = pd.read_csv(path, dtype={'customer_id': str})
    required = {'customer_id', 'segment_name', 'is_premium'}
    assert required.issubset(seg.columns), 'Missing required segment columns'
    assert seg.customer_id.is_unique, 'Duplicate customer_id in segment output'
    seg['is_premium'] = seg['is_premium'].astype(str).str.strip().str.lower().eq('true')
    return seg.set_index('customer_id')[['segment_name', 'is_premium']].to_dict('index')


def run_pipeline(prompt=FEW, token_budget=None, output_path=None):
    """Process tickets in original CSV order, reset mock before every full run.

    No budget: full run. Budget check is performed BEFORE each model call.
    The mock's usage.total_tokens is charged only for successful responses.
    """
    tickets = pd.read_csv(RAW / 'tickets.csv', keep_default_na=False, dtype={'ticket_id': str, 'customer_id': str, 'order_id': str})
    required = {'ticket_id', 'customer_id', 'order_id', 'text', 'true_category'}
    assert required.issubset(tickets.columns)
    segmap = _segments()
    privacy = audit_privacy(tickets)
    mock_llm.reset()
    random.seed(42)
    total_tokens = 0
    rows = []
    retries = []
    for ticket in tickets.itertuples(index=False):
        text = str(ticket.text)
        category, status, tokens = '', '', 0
        if not text.strip():
            status = 'skipped_empty'
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = 'flagged_long'
        else:
            safe, _ = redact(text)
            assert_redacted(safe)
            if token_budget is not None and total_tokens >= token_budget:
                status = 'budget_exhausted'
            else:
                messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': safe}]
                assert messages[-1]['content'] == safe
                assert_redacted(messages[-1]['content'])
                response = retry_chat(messages, ticket.ticket_id, retries)
                tokens = response.usage.total_tokens
                total_tokens += tokens
                category, status = validate_answer(response.choices[0].message.content)
        segment = segmap.get(ticket.customer_id, {})
        if status in {'skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted'}:
            route, priority = 'human_review', 'NORMAL'
        elif category == 'fraud':
            route, priority = 'security_team', 'HIGH'
        elif segment.get('is_premium', False):
            route, priority = 'premium_desk', 'HIGH'
        else:
            route, priority = 'support_queue', 'NORMAL'
        rows.append({
            'ticket_id': ticket.ticket_id, 'customer_id': ticket.customer_id,
            'status': status, 'category': category,
            'segment_name': segment.get('segment_name', '') or '',
            'priority': priority, 'route': route, 'tokens_used': tokens,
        })
    result = pd.DataFrame(rows, columns=[
        'ticket_id', 'customer_id', 'status', 'category',
        'segment_name', 'priority', 'route', 'tokens_used',
    ])
    classified = result.status.isin(['classified_json', 'classified_keyword'])
    true = tickets.loc[classified, 'true_category'].reset_index(drop=True)
    pred = result.loc[classified, 'category'].reset_index(drop=True)
    count = int(classified.sum())
    summary = {
        'tickets': len(result), 'classified': count,
        'correct': int((true == pred).sum()),
        'accuracy': float((true == pred).mean()) if count else 0.0,
        'tokens': int(total_tokens),
        'tokens_per_classified': total_tokens / count if count else None,
        'redactions': privacy,
        'retry_count': len(retries),
        'status_counts': result.status.value_counts().to_dict(),
        'route_counts': result.route.value_counts().to_dict(),
    }
    if output_path is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result, summary


def main():
    print('PART C: ZERO-SHOT FULL RUN')
    zero, zs = run_pipeline(prompt=ZERO)
    print(json.dumps(zs, indent=2))
    truth = pd.read_csv(RAW / 'tickets.csv', keep_default_na=False)['true_category']
    eligible = zero.status.isin(['classified_json', 'classified_keyword'])
    print('Zero-shot confusion table (true rows, predicted columns):')
    print(pd.crosstab(truth[eligible].reset_index(drop=True), zero.loc[eligible, 'category'].reset_index(drop=True), rownames=['true'], colnames=['predicted']))

    print('\nPART C: FEW-SHOT FULL RUN')
    few, fs = run_pipeline(prompt=FEW, token_budget=100000, output_path=OUT / 'triage_results.csv')
    print(json.dumps(fs, indent=2))

    print('\nPART C: ZERO-SHOT BUDGET DEMO')
    demo_budget = 6000
    demo, ds = run_pipeline(prompt=ZERO, token_budget=demo_budget)
    if ds['status_counts'].get('budget_exhausted', 0) == 0:
        demo_budget = 5000
        demo, ds = run_pipeline(prompt=ZERO, token_budget=demo_budget)
    demo.to_csv(OUT / 'triage_budget_demo.csv', index=False)
    print('budget:', demo_budget)
    print(json.dumps(ds, indent=2))
    assert ds['status_counts'].get('budget_exhausted', 0) > 0
    assert len(few) == 164
    assert zs['retry_count'] == 19 and fs['retry_count'] == 19
    assert fs['status_counts'].get('classified_json') == 135
    assert fs['status_counts'].get('classified_keyword') == 12
    assert fs['status_counts'].get('flagged_invalid_output') == 9
    assert fs['redactions'] == {'email': 55, 'phone': 91, 'order_id': 132, 'total': 278}
    print('\nPART C REFERENCE CHECKPOINTS PASSED')


if __name__ == '__main__':
    main()
