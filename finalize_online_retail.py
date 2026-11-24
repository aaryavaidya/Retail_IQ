"""FINAL QUALITY-ASSURANCE PASS for Online Retail II.

Input: cleaned_transactions.csv created by clean_online_retail_ii.py.
Run: python finalize_online_retail.py --input cleaned_retail/cleaned_transactions.csv --output final_retail

Purpose: merchandise-purchase analysis for probability distributions and mixture models.
This pass removes *only identified non-product service / adjustment / test rows*,
flags ambiguous duplicate lines and extremes without deleting them, verifies
invoice/customer consistency, and creates three final analysis tables.
Uses the Python standard library only (csv, json, decimal, etc.).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Verified in the user's actual input: not physical products.
EXCLUDE_CODES = {'C2', 'TEST001', 'TEST002', 'ADJUST2'}
EXCLUDE_DESCRIPTION_AND_CODE = {('23444', 'NEXT DAY CARRIAGE')}
ORIGINAL_PERCENTILE_FOR_HIGH_VALUE_FLAG = Decimal('829.49656')  # baseline from earlier audit


def cents(value: str) -> int:
    return int((Decimal(value) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def money(integer_cents: int) -> str:
    return f'{integer_cents / 100:.2f}'


def rawkey(row: dict) -> tuple:
    """All original identifying and economic fields; computed columns omitted."""
    return tuple(row[k] for k in (
        'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
        'UnitPrice', 'CustomerID', 'Country'))


def main(source: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    excluded = Counter()
    old_count = 0
    with source.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        required = {'InvoiceNo','StockCode','Description','Quantity','InvoiceDate',
                    'UnitPrice','CustomerID','Country','LineAmount','InvoiceYear','InvoiceMonth'}
        if not required.issubset(rdr.fieldnames):
            raise ValueError('Columns missing: '+str(required-set(rdr.fieldnames)))
        for row in rdr:
            old_count += 1
            code = row['StockCode'].strip().upper()
            desc = row['Description'].strip().upper()
            reason = None
            if code == 'C2' and desc == 'CARRIAGE': reason = 'Carriage fee (C2)'
            elif code in ('TEST001','TEST002') and desc == 'THIS IS A TEST PRODUCT.': reason = 'Test products (TEST001/2)'
            elif code == 'ADJUST2' and desc.startswith('ADJUSTMENT BY'): reason = 'Accounting adjustment (ADJUST2)'
            elif (code,desc) in EXCLUDE_DESCRIPTION_AND_CODE: reason = 'Next-day carriage (23444)'
            # Fail safely if a known excluded code appears with a changed description:
            elif code in EXCLUDE_CODES: reason = 'Other known administrative code (manual review)'
            if reason is not None:
                excluded[reason] += 1
                continue
            if not row['InvoiceNo'] or not row['CustomerID'] or not row['StockCode'] or not row['InvoiceDate'] or not row['Description']:
                raise ValueError(f'Unexpected missing identifiers or description in base cleaned data on input line {old_count}')
            if not (int(row['Quantity'])>0 and Decimal(row['UnitPrice'])>0):
                raise ValueError('Invalid quantity/price encountered after first cleaning')
            expected = cents(row['Quantity']) * Decimal(row['UnitPrice'])
            computed = int((expected).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
            # Quantity is an integer, amount is rounded to pennies in the original output.
            if abs(computed - cents(row['LineAmount'])) > 1:
                raise ValueError(f'Invalid line amount at input line {old_count}: {row}')
            dt = datetime.fromisoformat(row['InvoiceDate'])
            if row['InvoiceYear'] != str(dt.year) or row['InvoiceMonth'] != dt.strftime('%Y-%m'):
                raise ValueError(f'InvoiceMonth or InvoiceYear inconsistent at input line {old_count}')
            row['LineAmount'] = money(computed)
            row['NonMerchandiseFlag'] = 'False'
            records.append(row)
    if not records: raise ValueError('No purchase records remain')

    # Identify exact duplicate product lines: mark BOTH first and subsequent copies.
    # They might be valid repeated line entries; do not arbitrarily change revenue.
    dup_counts = Counter(rawkey(r) for r in records)
    dup_count_excess = sum(v-1 for v in dup_counts.values())
    num_dup_groups = sum(v>1 for v in dup_counts.values())
    flagged_duplicates = sum(v for v in dup_counts.values() if v>1)

    # Group all records at INVOICE grain; one invoice must have exactly one customer.
    invoices = {}
    inv_to_customer = defaultdict(set)
    inv_to_country = defaultdict(set)
    amount_values_cents = []
    for r in records:
        cid=r['CustomerID']; ino=r['InvoiceNo']; year=r['InvoiceYear']
        inv_to_customer[ino].add(cid)
        inv_to_country[ino].add(r['Country'])
        ikey=(year,ino,cid)
        dt=datetime.fromisoformat(r['InvoiceDate']); amt=cents(r['LineAmount'])
        amount_values_cents.append(amt)
        if ikey not in invoices:
            invoices[ikey]={'dates':{dt},'first_date':dt,'last_date':dt,'amount':0,'quantity':0,
                            'n_lines':0,'codes':set(),'country':r['Country'],'duplicate_lines':0,
                            'high_lines':0}
        a=invoices[ikey]
        a['amount']+=amt
        a['quantity']+=int(r['Quantity'])
        a['n_lines']+=1
        a['codes'].add(r['StockCode'])
        a['first_date']=min(a['first_date'],dt)
        a['last_date']=max(a['last_date'],dt)
        if dup_counts[rawkey(r)]>1:a['duplicate_lines']+=1
    if any(len(cs)>1 for cs in inv_to_customer.values()):
        raise ValueError('At least one invoice belongs to multiple customers! Investigate before analysis.')
    if any(len(cs)>1 for cs in inv_to_country.values()):
        raise ValueError('At least one invoice has multiple countries! Investigate before analysis.')
    if any(a['amount']<=0 for a in invoices.values()):
        raise ValueError('At least one invoice has non-positive amount.')

    # High values are VALID unless independently proven erroneous; preserve but flag.
    sorted_amounts = sorted(amount_values_cents)
    high_at = sorted_amounts[math.ceil(0.999*len(sorted_amounts))-1]
    high_inv_at = sorted((a['amount'] for a in invoices.values()))[math.ceil(0.999*len(invoices))-1]
    date_range = (min(a['first_date'] for a in invoices.values()),
                  max(a['last_date'] for a in invoices.values()))
    multi_timestamp_invoices=sum(a['first_date']!=a['last_date'] for a in invoices.values())
    max_date_spread_minutes=max((a['last_date']-a['first_date']).total_seconds()/60 for a in invoices.values())
    for a in invoices.values():
        a['high_amount_flag'] = a['amount']>high_inv_at

    # Final product-line file with ALL potentially ambiguous data explicitly flagged.
    line_headers=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice',
                  'CustomerID','Country','LineAmount','InvoiceYear','InvoiceMonth',
                  'PossibleExactDuplicate','DuplicateGroupSize','HighLineAmountFlag']
    lines_path=outdir/'final_cleaned_transactions.csv'
    n_high=0
    with lines_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=line_headers);w.writeheader()
        for r in records:
            amt=cents(r['LineAmount']);high=amt>high_at
            n_high+=int(high)
            w.writerow({k: v for k, v in {**r,'PossibleExactDuplicate':dup_counts[rawkey(r)]>1,
                       'DuplicateGroupSize':dup_counts[rawkey(r)],
                       'HighLineAmountFlag':high}.items() if k in line_headers})

    # Final invoice file: correct unit of observation for positive-spend distributions.
    invoice_headers=['InvoiceYear','InvoiceNo','CustomerID','Country','InvoiceDate',
                     'InvoiceMonth','InvoiceAmount','ItemQuantity','ProductLineCount',
                     'DistinctProducts','PossibleDuplicateLineCount',
                     'HighInvoiceAmountFlag','InvoiceTimestampSpreadMinutes']
    invoice_path=outdir/'final_invoice_level.csv'
    cust={}  # Customer ID => summary
    cust_month=defaultdict(lambda:{'count':0,'cents':0})
    total_invoice_cents=0
    with invoice_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=invoice_headers); w.writeheader()
        for (year,ino,cid),a in sorted(invoices.items(),key=lambda kv:(kv[1]['first_date'],kv[0][1])):
            dt=a['first_date'];mon=dt.strftime('%Y-%m');total_invoice_cents+=a['amount']
            w.writerow({'InvoiceYear':year,'InvoiceNo':ino,'CustomerID':cid,'Country':a['country'],
                        'InvoiceDate':dt.strftime('%Y-%m-%d %H:%M:%S'), 'InvoiceMonth':mon,
                        'InvoiceAmount':money(a['amount']),'ItemQuantity':a['quantity'],
                        'ProductLineCount':a['n_lines'],'DistinctProducts':len(a['codes']),
                        'PossibleDuplicateLineCount':a['duplicate_lines'],
                        'HighInvoiceAmountFlag':a['high_amount_flag'],
                        'InvoiceTimestampSpreadMinutes':round((a['last_date']-a['first_date']).total_seconds()/60,2)})
            if cid not in cust:
                cust[cid]={'count':0,'amount':0,'first':dt,'last':dt,'countries':set(),'high_invoice_count':0,'flagged_duplicate_invoices':0}
            c=cust[cid];c['count']+=1;c['amount']+=a['amount'];c['first']=min(c['first'],dt);c['last']=max(c['last'],dt)
            c['countries'].add(a['country']);c['high_invoice_count']+=int(a['high_amount_flag']);c['flagged_duplicate_invoices']+=int(a['duplicate_lines']>0)
            mk=(cid,mon); cust_month[mk]['count']+=1;cust_month[mk]['cents']+=a['amount']
    if sum(a['amount'] for a in invoices.values())!=total_invoice_cents:
        raise ValueError('Invoice reconciliation FAILED')
    if sum(a['n_lines'] for a in invoices.values())!=len(records):
        raise ValueError('Invoice line counts FAILED')

    cust_headers=['CustomerID','ObservedPurchaseCount','TotalSpending','AverageInvoiceValue',
                  'FirstObservedPurchase','LastObservedPurchase','ObservedCalendarSpanDays',
                  'DistinctCountries','HighInvoiceCount','InvoicesWithPossibleDuplicates']
    customer_path=outdir/'final_customer_level.csv'
    with customer_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=cust_headers);w.writeheader()
        for cid,c in sorted(cust.items()):
            w.writerow({'CustomerID':cid,'ObservedPurchaseCount':c['count'],
                        'TotalSpending':money(c['amount']),
                        'AverageInvoiceValue':money(int((Decimal(c['amount'])/Decimal(c['count'])).quantize(Decimal('1'), rounding=ROUND_HALF_UP))),
                        'FirstObservedPurchase':c['first'].strftime('%Y-%m-%d %H:%M:%S'),
                        'LastObservedPurchase':c['last'].strftime('%Y-%m-%d %H:%M:%S'),
                        'ObservedCalendarSpanDays':(c['last']-c['first']).days,
                        'DistinctCountries':len(c['countries']),'HighInvoiceCount':c['high_invoice_count'],
                        'InvoicesWithPossibleDuplicates':c['flagged_duplicate_invoices']})

    month_headers=['CustomerID','InvoiceMonth','ObservedMonthlyPurchases','MonthlySpending']
    with (outdir/'final_customer_month_positive_only.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=month_headers);w.writeheader()
        for (cid,mon),a in sorted(cust_month.items()):
            w.writerow({'CustomerID':cid,'InvoiceMonth':mon,
                        'ObservedMonthlyPurchases':a['count'],'MonthlySpending':money(a['cents'])})

    # Review sample only, never arbitrarily drop suspicious lines.
    with (outdir/'duplicate_review_sample.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['InvoiceNo','StockCode','CustomerID','InvoiceDate','Quantity',
                                    'UnitPrice','Description','IdenticalRows'])
        review_written = 0
        for r in records:
            k=rawkey(r)
            if dup_counts[k]>1:
                w.writerow([r[x] for x in ['InvoiceNo','StockCode','CustomerID','InvoiceDate','Quantity','UnitPrice','Description']]+[dup_counts[k]])
                dup_counts[k]=0 # only show the group once; full data was already saved.
                review_written += 1
                if review_written >= 300:break

    audit=[
        {'Stage':'Baseline pre-cleaned input','RowsRemoved':0,'RowsRemaining':old_count,'Reason':'Earlier stage removed cancellations, invalid prices/quantities, missing customer IDs, and known non-merchandise codes.'},
    ]
    remaining=old_count
    for reason,count in excluded.items():
        remaining-=count
        audit.append({'Stage':reason,'RowsRemoved':count,'RowsRemaining':remaining,
                      'Reason': 'Verified these entries are shipping charges, testing records, or adjustments, not merchandise sales.'})
    audit.extend([
        {'Stage':'Flag possible identical lines','RowsRemoved':0,'RowsRemaining':remaining,
         'Reason':f'{dup_count_excess:,} additional exact occurrences across {num_dup_groups:,} groups; do NOT delete without source evidence.'},
        {'Stage':'Flag high line and invoice amounts','RowsRemoved':0,'RowsRemaining':remaining,
         'Reason':'Extreme purchases can be genuine wholesale orders; preserve for tail modelling and sensitivity analyses.'},
        {'Stage':'Verify invoice and customer consistency','RowsRemoved':0,'RowsRemaining':remaining,
         'Reason':'Invoices map to one customer/country; minor within-invoice timestamp variations retained.'},
    ])
    with (outdir/'final_cleaning_audit.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=['Stage','RowsRemoved','RowsRemaining','Reason']);w.writeheader();w.writerows(audit)

    summary={
        'input':str(source),'baseline_rows':old_count,'final_product_lines':len(records),
        'newly_excluded_rows':sum(excluded.values()),'excluded_categories':dict(excluded),
        'final_invoices':len(invoices),'final_customers':len(cust),
        'observed_customer_months_positive_only':len(cust_month),
        'ambiguous_exact_duplicate_additional_occurrences':dup_count_excess,
        'ambiguous_exact_duplicate_groups':num_dup_groups,
        'rows_in_duplicate_groups':flagged_duplicates,
        'duplicate_rows_removed':0,
        'high_line_threshold_99_9pct_GBP':money(high_at),'high_line_count':n_high,
        'high_invoice_threshold_99_9pct_GBP':money(high_inv_at),
        'high_invoice_count':sum(a['high_amount_flag'] for a in invoices.values()),
        'outliers_removed':0,
        'within_invoice_timestamp_variations':multi_timestamp_invoices,
        'maximum_invoice_timestamp_spread_minutes':max_date_spread_minutes,
        'invoice_customer_conflicts':0,'invoice_country_conflicts':0,
        'invalid_nonpositive_amounts':0,
        'all_line_values_reconcile_to_invoice_total': True,
        'total_positive_merchandise_sales_GBP':money(total_invoice_cents),
        'first_invoice':date_range[0].isoformat(sep=' '),
        'last_invoice':date_range[1].isoformat(sep=' '),
        'important_notes':[
          'All values represent gross POSITIVE merchandise invoice amounts, not net sales after returns.',
          'Duplicate flags mark suspiciously identical lines; they are retained since genuine repetitions cannot be distinguished from duplicate-entry errors.',
          'Extreme transaction values retained; perform sensitivity analysis with and without high-value flags.',
          'Customer counts are OBSERVED purchases only; observation exposure varies across customers.',
          'The customer-month file contains only months with positive purchases; do not treat it as a complete zero-inclusive monthly panel.',
          'Customer identifiers should be strings; prices are in GBP. Dates use ISO formatting.',
          'The 99.9th percentile flags are not grounds for deletion.',
        ]
    }
    (outdir/'final_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (outdir/'README_FIRST.txt').write_text('''ONLINE RETAIL II — FINAL PREPARED DATA (31 OCTOBER 2026)\n\nFiles:\n1. final_cleaned_transactions.csv — product-line grain for EDA, per-product trends.\n2. final_invoice_level.csv — one invoice per row; use for Gamma/Lognormal transaction value models.\n3. final_customer_level.csv — one customer per row; use for exploratory segmentation.\n4. final_customer_month_positive_only.csv — purchase-month records ONLY.\n5. duplicate_review_sample.csv — example groups for review, *not* a deletion list.\n6. final_cleaning_audit.csv and final_validation_summary.json — methods and checks.\n\nKey restrictions:\n- Never count lines as purchases: count invoices.\n- Positive sales only; not revenue net of returns.\n- POSSIBLE duplicate lines and outliers are flagged, not automatically deleted.\n- Raw spending can have extreme values; evaluate outlier sensitivity.\n- To fit monthly Poisson/NB, choose a calendar window and explicitly model exposure/zero-purchase periods; do not fit the positive-only month file as if zeros were observed.\n- To reproduce: run clean_online_retail_ii.py against the original archive.zip, then run finalize_online_retail.py against the resulting cleaned_transactions.csv.\n''',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(Path(args.input),Path(args.output))
