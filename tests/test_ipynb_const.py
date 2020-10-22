from freezegun import freeze_time
import pandas  # pylint: disable=unused-import

# Note: The pandas import is required to avoid interference between pandas and
# freezegun magics to work.


def test_import_after_start():
    with freeze_time('2020-02-02'):
        from muttlib.ipynb_const import (
            END_DATE,
            LAST_WEEK_DATE,
            FRIENDS_DAY,
            KIDS_DAY,
            MOTHERS_DAY,
            FATHERS_DAY,
        )

        cases = [
            (END_DATE, '2020-02-02T00:00:00'),
            (LAST_WEEK_DATE, '2020-01-26T00:00:00'),
            (FRIENDS_DAY, '2020-07-20T00:00:00'),
            (KIDS_DAY, '2020-08-16T00:00:00'),
            (MOTHERS_DAY, '2020-10-18T00:00:00'),
            (FATHERS_DAY, '2020-06-21T00:00:00'),
        ]
        for dt, expected_dt in cases:
            assert dt.isoformat() == expected_dt
