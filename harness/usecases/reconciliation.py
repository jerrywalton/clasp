"""Ledger reconciliation: a trial balance folded from events alone."""

from __future__ import annotations

from dataclasses import dataclass

from core.events import DomainEvent
from core.ledger import AccountTotal, LedgerTransactionStatus, fold_ledger


@dataclass(frozen=True)
class TrialBalance:
    accounts: tuple[AccountTotal, ...]
    total_debits_cents: int
    total_credits_cents: int
    completed_debits_cents: int
    completed_credits_cents: int
    incomplete_transactions: tuple[LedgerTransactionStatus, ...]

    @property
    def completed_balanced(self) -> bool:
        return self.completed_debits_cents == self.completed_credits_cents

    @property
    def balanced(self) -> bool:
        """True when every observed transaction is balanced (which also
        makes the grand totals over all posted legs agree)."""
        return (not self.incomplete_transactions
                and self.total_debits_cents == self.total_credits_cents)


def trial_balance(events: list[DomainEvent]) -> TrialBalance:
    """Compute a trial balance from an event log alone.

    Per-account and grand totals cover every valid posted leg, so a
    half-posted transaction visibly unbalances the grand totals. The
    completed totals cover only balanced transactions and are the pair
    that must agree for the completed books.
    """
    book = fold_ledger(events)
    completed = [txn for txn in book.transactions if txn.balanced]
    return TrialBalance(
        accounts=tuple(book.accounts.values()),
        total_debits_cents=book.total_debits_cents,
        total_credits_cents=book.total_credits_cents,
        completed_debits_cents=sum(t.debit_cents for t in completed),
        completed_credits_cents=sum(t.credit_cents for t in completed),
        incomplete_transactions=book.incomplete)
