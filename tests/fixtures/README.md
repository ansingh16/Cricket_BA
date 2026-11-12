# Test fixtures

Four IPL match files from [Cricsheet](https://cricsheet.org/), unchanged, licensed under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
Each one covers a case the parser has to get right:

| File | Match | Case |
|---|---|---|
| `1136564.json` | 2018, Rajasthan Royals v Sunrisers Hyderabad | normal result; one over in the chase had 7 legal balls (`miscounted_overs`) |
| `1473469.json` | 2025, Delhi Capitals v Rajasthan Royals | tie decided by a super over (4 innings) |
| `1473495.json` | 2025, Punjab Kings v Delhi Capitals | no result, abandoned during the first innings |
| `335988.json` | 2008, Deccan Chargers v Delhi Daredevils | 5 penalty runs on a legal ball |
