# Global Warming Web App

## make_data
The python script `make_data.py` takes temperature and location data from 
* ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz
* ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
* https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/
and turns it into a usable [duckdb](https://duckdb.org/) database file which contains three tables
| Table | Description |
| :---  | :---        |
| loc_to_temp | Connects locations to a time series of yearly average temperatures |
| place_names | Connects names of places to latitude & longitude |
| place_zips  | Connects zip codes to latitude & longitude |

## server
`server.py` is the flask web app which takes the `database.duckdb` generated with `make_data.py` and 
makes it usable by people.

## Legal
All Code is Licensed under [MPLv2](https://www.mozilla.org/en-US/MPL/)