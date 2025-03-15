"""
Script for scraping player data from each team in each season from basketball reference.

This file requires basketball_reference_scraper. Instructions to instaling it is in README.
"""

from basketball_reference_scraper.teams import get_roster_stats

def main():
    team_abrv = [
        "ATL", "MIL", "TCB", "BOS", "BRK", "NJN", "CHI", "CHO", "CHA", 
        "CLE", "DAL", "DEN", "DET", "FWP", "GSW", "SFW", "PHI", "HOU", "IND", 
        "LAC", "SDC", "BUF", "LAL", "MIN", "MEM", "VAN", "MIA", "MIL", "MIN", 
        "NOP", "NOK", "NOH", "NYK", "OKC", "SEA", "ORL", "PHI", "SYR", "PHO", 
        "POR", "SAC", "KCK", "KCK", "CIN", "ROR", "SAS", "TOR", "UTA", "NOJ", 
        "WAS", "WAS", "CAP", "BAL", "CHI", "CHI", "AND", "CHI", "IND", "SRS", 
        "SLB", "WAS", "WAT"
    ]

    years = range(1979, 2025)

    for team in team_abrv:
        for yr in years:
            print(f"Team: {team}", f"Season: {yr-1}-{yr}")
            try:
                s = get_roster_stats(team, yr, data_format='PER_GAME', playoffs=False)
                s.to_csv(f"data/{yr}_{team}_roster_stats.csv")

            except Exception as e:
                print(f"Failed to get stats for {team} in {yr}: {e}")
                continue

    print("Data Extraction Complete")

if __name__ == "__main__":
    main()
