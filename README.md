# birthrate-factors-sparse-model-python

## Setup

To set up this project, use the following command:

```bash
git clone git@github.com:YMWjp/birthrate-factors-sparse-model-python.git
```

## Branching Strategy

When creating new features, please use the following naming convention for branches:

- `feat/~~` for new features
- `fix/~~` for bug fixes

## About Data

<details>
<summary>Click to expand data descriptions</summary>

### Description of each data

- `birth-rate-vs-death-rate.csv`:
  - [link](https://ourworldindata.org/grapher/birth-rate-vs-death-rate)
  - Birth rate vs. death rate, 2023
    - Rates are given per 1,000 people in the country's population. Countries above the gray line have a higher birth than death rate, meaning that the total population is increasing; those below the line have a declining population.
- `female-labor-force-participation-rates-by-national-per-capita-income.csv`:
  - [link](https://ourworldindata.org/grapher/female-labor-force-participation-rates-by-national-per-capita-income)
  - Female labor force participation rates by national per capita income, 2022
    - The labor force participation rate corresponds to the proportion of the population ages 15 and older that is economically active. National income levels correspond to GDP per capita in constant international dollars. This means figures are adjusted for inflation and cross-country price differences.
- `human-development-index.csv`:
  - [link](https://ourworldindata.org/grapher/human-development-index)
  - Human Development Index, 2022
    - The Human Development Index (HDI) is a summary measure of key dimensions of human development: a long and healthy life, a good education, and a decent standard of living. Higher values indicate higher human development.
- `maddison-data-gdp-per-capita-in-2011us-slopechart.csv`:
  - [link](https://ourworldindata.org/grapher/maddison-data-gdp-per-capita-in-2011us-slopechart)
  - GDP per capita, 1950 to 2022
    - This data is adjusted for inflation and for differences in the cost of living between countries.
- `share-of-population-urban.csv`:
  - [link](https://ourworldindata.org/grapher/share-of-population-urban)
  - Share of the population living in urban areas, 2022
- `physicians-per-1000-people.csv`:

  - [link](https://ourworldindata.org/grapher/physicians-per-1000-people)
  - Medical doctors per 1,000 people, 2021

  https://ourworldindata.org/grapher/urban-share-european-commission

  https://ourworldindata.org/grapher/suicide-rate-who-mdb

  https://ourworldindata.org/grapher/migrant-stock-total

  https://ourworldindata.org/grapher/children-per-woman-un

  https://ourworldindata.org/working-hours

  https://ourworldindata.org/grapher/minutes-spent-on-leisure

  https://ourworldindata.org/age-structure

</details>

## Directory Structure

This project adopts the following directory structure to clearly separate the functionalities of data processing, analysis, and visualization:

- `data/`: A folder for storing raw and processed data.
  - `raw/`: subfolder contains the raw data.
  - `processed/`: subfolder holds the preprocessed data.
- `src/`: A folder for storing the source code.
  - `data_processing.py`: for data processing.
  - `analysis.py`: for data analysis.
  - `visualization.py`: for data visualization.
  - `main.py`: for managing the overall workflow.
- `requirements.txt`: A file listing the necessary Python packages for the project.
- `.gitignore`: A file specifying which files should be ignored by Git.
