#!/bin/bash
# Fetch data from May 2025 to Dec 2025 (8 months) + Jan 2026 if available.
# Note: Binanc Vision often has a lag, so Jan 2026 might not be there yet, but we'll try.

YEARS=("2025" "2026")

for year in "${YEARS[@]}"; do
	if [ "$year" == "2025" ]; then
		MONTHS=("02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
	else
		MONTHS=("01")
	fi

	for month in "${MONTHS[@]}"; do
		echo "Fetching $year-$month..."
		.venv/bin/python data_fetchers/full_data_fetcher.py --year "$year" --month "$month"
	done
done
