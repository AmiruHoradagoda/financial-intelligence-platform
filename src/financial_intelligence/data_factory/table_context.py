"""Small deterministic context reader for two-column annual-report tables."""

import re
from decimal import Decimal


def _number(value: str) -> str:
    return str(Decimal(value.replace(",", "")).normalize())


def table_support(
    text: str, quotes: list[str], question: str, response: str
) -> tuple[set[str], set[str], str | None]:
    """Return supported header numbers/units and any row/year contradiction.

    Only rows present verbatim in supplied evidence are used. Headers supply
    context, never permission to borrow amounts from unquoted rows.
    """
    header = re.search(r"^\s*((?:20\d{2}\s+){1,3}20\d{2})\s*$", text, re.M)
    if not header:
        return set(), set(), None
    years = re.findall(r"20\d{2}", header.group(1))
    if len(years) != 2 or len(set(years)) != 2:
        return set(), set(), None
    amount = r"\d[\d,]*(?:\.\d+)?"
    row_pattern = re.compile(
        rf"^\s*(.+?[A-Za-z].*?)\s{{2,}}\$?\s*({amount})\s+\$?\s*({amount})\s*$"
    )
    rows = []
    for line in text[header.end() :].splitlines():
        match = row_pattern.fullmatch(line)
        if match:
            rows.append(
                (
                    match[1].strip().casefold(),
                    dict(zip(years, (match[2], match[3]))),
                    line.strip(),
                )
            )
    referenced = [row for row in rows if any(row[2] in quote for quote in quotes)]
    if not referenced:
        return set(), set(), None
    question = " ".join(question.casefold().split())
    subjects = [row for row in rows if row[0] in question]
    if subjects and any(row not in referenced for row in subjects):
        return set(years), set(), "table_row_mismatch"
    if len(subjects) > 1:
        response_numbers = {_number(n) for n in re.findall(amount, response)}
        if any(
            row[0] not in response.casefold()
            and not response_numbers.intersection(
                {_number(value) for value in row[1].values()}
            )
            for row in subjects
        ):
            return set(years), set(), "table_row_coverage_mismatch"
    units_match = re.search(
        r"\(in\s+(thousands|millions|billions)\)", text[: header.start()], re.I
    )
    units = {units_match[1].casefold().rstrip("s")} if units_match else set()
    claimed_units = set(
        re.findall(r"\b(thousand|million|billion|trillion)s?\b", response.casefold())
    )
    if claimed_units and units and not claimed_units <= units:
        return set(years), units, "table_unit_mismatch"
    # A direct one-year question must use that year's cell, not just a value
    # found somewhere in the row. Including the other year's cell is flagged too.
    requested_years = set(re.findall(r"\b20\d{2}\b", question))
    if len(subjects) == 1 and len(requested_years) == 1:
        year = next(iter(requested_years))
        amounts = {_number(n) for n in re.findall(amount, response) if n not in years}
        expected = subjects[0][1].get(year)
        if expected is None or not amounts or amounts != {_number(expected)}:
            return set(years), units, "table_year_value_mismatch"
    # Explicit comparative bindings: '$4,356 million in 2023' or '2024: $4,337'.
    before = re.compile(
        rf"\$?({amount})\s*(?:(?:million|billion|thousand)s?\s*)?(?:in|for|during)\s+(20\d{{2}})\b",
        re.I,
    )
    after = re.compile(
        rf"\b(20\d{{2}})\b(?:(?!20\d{{2}})[^.;\n]){{0,100}}?(?::|=|\bwas|\bwere|\btotaled)\s*\$?({amount})",
        re.I,
    )
    for sentence in re.split(r"[\n;]|(?<=[.!?])\s+", response):
        local = [row for row in referenced if row[0] in sentence.casefold()]
        if not local and len(subjects) == 1:
            local = subjects
        if len(local) != 1:
            continue
        row = local[0]
        bindings = [(year, value) for value, year in before.findall(sentence)]
        bindings += after.findall(sentence)
        for year, value in bindings:
            if year not in row[1] or _number(value) != _number(row[1][year]):
                return set(years), units, "table_year_value_mismatch"
        if requested_years == set(years):
            difference = Decimal(row[1][years[1]].replace(",", "")) - Decimal(
                row[1][years[0]].replace(",", "")
            )
            if difference < 0 and re.search(
                r"\b(increased|rose|grew)\b", sentence, re.I
            ):
                return set(years), units, "table_change_direction_mismatch"
            if difference > 0 and re.search(
                r"\b(decreased|fell|declined)\b", sentence, re.I
            ):
                return set(years), units, "table_change_direction_mismatch"
    return set(years), units, None
