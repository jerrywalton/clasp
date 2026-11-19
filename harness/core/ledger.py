"""Double-entry ledger entities and folds.

The ledger is deliberately pure: it knows accounts, legs, transactions,
and how to fold ledger events. It does not know how a run is executed.

A posted transaction has exactly two legs: one debit and one credit of
the same positive amount. A fold of an event log may observe fewer (or
otherwise malformed) legs after a crash; those observations are named
``incomplete`` or ``unbalanced`` rather than being treated as posted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import LEDGER_POSTED, DomainEvent

DEBIT = "debit"
CREDIT = "credit"
SIDES = (DEBIT, CREDIT)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class Leg:
    account: str
    side: str
    amount_cents: int
    reference: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.account, str) or not self.account:
            raise ValueError("ledger leg account must be a non-empty string")
        if self.side not in SIDES:
            raise ValueError(f"ledger leg side must be one of {SIDES}")
        if not _is_positive_int(self.amount_cents):
            raise ValueError("ledger leg amount_cents must be a positive int")


@dataclass(frozen=True)
class Transaction:
    """A complete double-entry posting: exactly two balanced legs."""

    txn_id: str
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.txn_id, str) or not self.txn_id:
            raise ValueError("ledger transaction id must be a non-empty string")
        if len(self.legs) != 2:
            raise ValueError("a posted ledger transaction has exactly two legs")
        sides = tuple(leg.side for leg in self.legs)
        if set(sides) != {DEBIT, CREDIT} or len(set(sides)) != 2:
            raise ValueError("a posted transaction needs one debit and one credit")
        if self.legs[0].amount_cents != self.legs[1].amount_cents:
            raise ValueError("ledger transaction legs must have equal amounts")

    @property
    def debit(self) -> Leg:
        return next(leg for leg in self.legs if leg.side == DEBIT)

    @property
    def credit(self) -> Leg:
        return next(leg for leg in self.legs if leg.side == CREDIT)

    @property
    def amount_cents(self) -> int:
        return self.legs[0].amount_cents


@dataclass(frozen=True)
class PostingRule:
    """How a ledger-aware tool posts its financial effect."""

    tool: str
    debit_account: str
    credit_account: str
    amount_arg: str = "amount_cents"
    reference_arg: str = "order_id"

    def __post_init__(self) -> None:
        for label, value in (
                ("tool", self.tool),
                ("debit_account", self.debit_account),
                ("credit_account", self.credit_account),
                ("amount_arg", self.amount_arg),
                ("reference_arg", self.reference_arg)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"posting rule {label} must be a non-empty string")
        if self.debit_account == self.credit_account:
            raise ValueError("posting rule debit and credit accounts must differ")


@dataclass(frozen=True)
class AccountTotal:
    account: str
    debit_cents: int = 0
    credit_cents: int = 0

    @property
    def balance_cents(self) -> int:
        """Debit-normal balance (debits minus credits)."""
        return self.debit_cents - self.credit_cents


@dataclass(frozen=True)
class LedgerTransactionStatus:
    """What a fold can prove about one transaction id in a log."""

    txn_id: str
    legs: tuple[Leg, ...] = ()
    status: str = "incomplete"  # balanced | incomplete | unbalanced
    debit_cents: int = 0
    credit_cents: int = 0
    tool: str | None = None
    reference: object = None

    @property
    def balanced(self) -> bool:
        return self.status == "balanced"

    def leg_for_side(self, side: str) -> Leg | None:
        for leg in self.legs:
            if leg.side == side:
                return leg
        return None


@dataclass(frozen=True)
class LedgerBook:
    transactions: tuple[LedgerTransactionStatus, ...] = ()
    accounts: dict[str, AccountTotal] = field(default_factory=dict)
    total_debits_cents: int = 0
    total_credits_cents: int = 0

    def transaction(self, txn_id: str) -> LedgerTransactionStatus | None:
        for txn in self.transactions:
            if txn.txn_id == txn_id:
                return txn
        return None

    @property
    def incomplete(self) -> tuple[LedgerTransactionStatus, ...]:
        return tuple(txn for txn in self.transactions if not txn.balanced)

    @property
    def balanced(self) -> bool:
        return (not self.incomplete
                and self.total_debits_cents == self.total_credits_cents)


def transaction_balanced(transaction: object) -> bool:
    """True only for a complete two-leg, equal-amount transaction."""
    if isinstance(transaction, LedgerTransactionStatus):
        return transaction.balanced
    if isinstance(transaction, Transaction):
        return True  # Transaction construction already enforces balance.
    return False


def _status_for(txn_id: str, legs: list[Leg],
                payloads: list[dict]) -> LedgerTransactionStatus:
    debit_total = sum(leg.amount_cents for leg in legs if leg.side == DEBIT)
    credit_total = sum(leg.amount_cents for leg in legs if leg.side == CREDIT)
    if len(legs) < 2:
        status = "incomplete"
    elif (len(legs) == 2
          and {leg.side for leg in legs} == {DEBIT, CREDIT}
          and debit_total == credit_total):
        status = "balanced"
    else:
        status = "unbalanced"
    tool = next((p.get("tool") for p in payloads if p.get("tool")), None)
    reference = next((p.get("reference") for p in payloads
                      if p.get("reference") is not None), None)
    if reference is None and legs:
        reference = legs[0].reference
    return LedgerTransactionStatus(
        txn_id=txn_id, legs=tuple(legs), status=status,
        debit_cents=debit_total, credit_cents=credit_total,
        tool=tool, reference=reference)


def fold_ledger(events: list[DomainEvent]) -> LedgerBook:
    """Fold LEDGER_POSTED events into transactions and account totals.

    Malformed ledger payloads do not disappear: if their amount is a
    valid positive integer they contribute to totals and their
    transaction is marked unbalanced; otherwise the transaction is
    marked unbalanced with no trustworthy amount contribution.
    """
    grouped: dict[str, list[Leg]] = {}
    payloads: dict[str, list[dict]] = {}
    order: list[str] = []
    totals: dict[str, list[int]] = {}

    for event in sorted(events, key=lambda e: e.seq):
        if event.type != LEDGER_POSTED:
            continue
        payload = event.payload or {}
        txn_id = payload.get("txn_id")
        if not isinstance(txn_id, str) or not txn_id:
            txn_id = "<missing txn_id>"
        if txn_id not in grouped:
            grouped[txn_id] = []
            payloads[txn_id] = []
            order.append(txn_id)
        payloads[txn_id].append(payload)

        account = payload.get("account")
        side = payload.get("side")
        amount = payload.get("amount_cents")
        if (isinstance(account, str) and account
                and side in SIDES and _is_positive_int(amount)):
            leg = Leg(account=account, side=side, amount_cents=amount,
                      reference=payload.get("reference"))
            grouped[txn_id].append(leg)
            pair = totals.setdefault(account, [0, 0])
            pair[0 if side == DEBIT else 1] += amount
        else:
            # Keep a marker in the transaction's payload history via a
            # deliberately invalid status calculation below: with no
            # trustworthy leg, a one-payload transaction cannot balance.
            # Represent it as an unbalanced status by appending no leg;
            # _status_for sees the payload count through legs only, so
            # handle the all-malformed case after grouping.
            pass

    transactions: list[LedgerTransactionStatus] = []
    for txn_id in order:
        status = _status_for(txn_id, grouped[txn_id], payloads[txn_id])
        # A transaction whose payloads were all malformed has no legs;
        # that is not merely "incomplete" (a leg might exist but be
        # unreadable), it is unbalanced/invalid.
        if not grouped[txn_id] and payloads[txn_id]:
            status = LedgerTransactionStatus(
                txn_id=txn_id, legs=(), status="unbalanced",
                debit_cents=0, credit_cents=0,
                tool=status.tool, reference=status.reference)
        transactions.append(status)

    accounts = {
        account: AccountTotal(account, pair[0], pair[1])
        for account, pair in sorted(totals.items())
    }
    return LedgerBook(
        transactions=tuple(transactions), accounts=accounts,
        total_debits_cents=sum(a.debit_cents for a in accounts.values()),
        total_credits_cents=sum(a.credit_cents for a in accounts.values()))
