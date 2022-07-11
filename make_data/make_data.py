import os
import shutil
import tarfile
import urllib.request as request
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import duckdb as ddb
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
ID         is the station identification code.  Please see "ghcnd-stations.txt"
           for a complete list of stations and their metadata.
YEAR       is the year of the record.

MONTH      is the month of the record.

ELEMENT    is the element type.   There are five core elements as well as a number
           of addition elements.  
	   
	   The five core elements are:

           PRCP = Precipitation (tenths of mm)
   	   SNOW = Snowfall (mm)
	   SNWD = Snow depth (mm)
           TMAX = Maximum temperature (tenths of degrees C)
           TMIN = Minimum temperature (tenths of degrees C)
	   
	   The other elements are:
	   
	   ACMC = Average cloudiness midnight to midnight from 30-second 
	          ceilometer data (percent)
	   ACMH = Average cloudiness midnight to midnight from 
	          manual observations (percent)
           ACSC = Average cloudiness sunrise to sunset from 30-second 
	          ceilometer data (percent)
	   ACSH = Average cloudiness sunrise to sunset from manual 
	          observations (percent)
           AWDR = Average daily wind direction (degrees)
	   AWND = Average daily wind speed (tenths of meters per second)
	   DAEV = Number of days included in the multiday evaporation
	          total (MDEV)
	   DAPR = Number of days included in the multiday precipiation 
	          total (MDPR)
           DASF = Number of days included in the multiday snowfall 
	          total (MDSF)		  
	   DATN = Number of days included in the multiday minimum temperature 
	         (MDTN)
	   DATX = Number of days included in the multiday maximum temperature 
	          (MDTX)
           DAWM = Number of days included in the multiday wind movement
	          (MDWM)
	   DWPR = Number of days with non-zero precipitation included in 
	          multiday precipitation total (MDPR)
	   EVAP = Evaporation of water from evaporation pan (tenths of mm)
	   FMTM = Time of fastest mile or fastest 1-minute wind 
	          (hours and minutes, i.e., HHMM)
	   FRGB = Base of frozen ground layer (cm)
	   FRGT = Top of frozen ground layer (cm)
	   FRTH = Thickness of frozen ground layer (cm)
	   GAHT = Difference between river and gauge height (cm)
	   MDEV = Multiday evaporation total (tenths of mm; use with DAEV)
	   MDPR = Multiday precipitation total (tenths of mm; use with DAPR and 
	          DWPR, if available)
	   MDSF = Multiday snowfall total 
	   MDTN = Multiday minimum temperature (tenths of degrees C; use with 
	          DATN)
	   MDTX = Multiday maximum temperature (tenths of degress C; use with 
	          DATX)
	   MDWM = Multiday wind movement (km)
           MNPN = Daily minimum temperature of water in an evaporation pan 
	         (tenths of degrees C)
           MXPN = Daily maximum temperature of water in an evaporation pan 
	         (tenths of degrees C)
	   PGTM = Peak gust time (hours and minutes, i.e., HHMM)
	   PSUN = Daily percent of possible sunshine (percent)
	   SN*# = Minimum soil temperature (tenths of degrees C)
	          where * corresponds to a code
	          for ground cover and # corresponds to a code for soil 
		  depth.  
		  
		  Ground cover codes include the following:
		  0 = unknown
		  1 = grass
		  2 = fallow
		  3 = bare ground
		  4 = brome grass
		  5 = sod
		  6 = straw multch
		  7 = grass muck
		  8 = bare muck
		  
		  Depth codes include the following:
		  1 = 5 cm
		  2 = 10 cm
		  3 = 20 cm
		  4 = 50 cm
		  5 = 100 cm
		  6 = 150 cm
		  7 = 180 cm
		  
	   SX*# = Maximum soil temperature (tenths of degrees C) 
	          where * corresponds to a code for ground cover 
		  and # corresponds to a code for soil depth. 
		  See SN*# for ground cover and depth codes. 
           TAVG = Average temperature (tenths of degrees C)
	          [Note that TAVG from source 'S' corresponds
		   to an average for the period ending at
		   2400 UTC rather than local midnight]
           THIC = Thickness of ice on water (tenths of mm)	
 	   TOBS = Temperature at the time of observation (tenths of degrees C)
	   TSUN = Daily total sunshine (minutes)
	   WDF1 = Direction of fastest 1-minute wind (degrees)
	   WDF2 = Direction of fastest 2-minute wind (degrees)
	   WDF5 = Direction of fastest 5-second wind (degrees)
	   WDFG = Direction of peak wind gust (degrees)
	   WDFI = Direction of highest instantaneous wind (degrees)
	   WDFM = Fastest mile wind direction (degrees)
           WDMV = 24-hour wind movement (km)	   
           WESD = Water equivalent of snow on the ground (tenths of mm)
	   WESF = Water equivalent of snowfall (tenths of mm)
	   WSF1 = Fastest 1-minute wind speed (tenths of meters per second)
	   WSF2 = Fastest 2-minute wind speed (tenths of meters per second)
	   WSF5 = Fastest 5-second wind speed (tenths of meters per second)
	   WSFG = Peak gust wind speed (tenths of meters per second)
	   WSFI = Highest instantaneous wind speed (tenths of meters per second)
	   WSFM = Fastest mile wind speed (tenths of meters per second)
	   WT** = Weather Type where ** has one of the following values:
	   
                  01 = Fog, ice fog, or freezing fog (may include heavy fog)
                  02 = Heavy fog or heaving freezing fog (not always 
		       distinquished from fog)
                  03 = Thunder
                  04 = Ice pellets, sleet, snow pellets, or small hail 
                  05 = Hail (may include small hail)
                  06 = Glaze or rime 
                  07 = Dust, volcanic ash, blowing dust, blowing sand, or 
		       blowing obstruction
                  08 = Smoke or haze 
                  09 = Blowing or drifting snow
                  10 = Tornado, waterspout, or funnel cloud 
                  11 = High or damaging winds
                  12 = Blowing spray
                  13 = Mist
                  14 = Drizzle
                  15 = Freezing drizzle 
                  16 = Rain (may include freezing rain, drizzle, and
		       freezing drizzle) 
                  17 = Freezing rain 
                  18 = Snow, snow pellets, snow grains, or ice crystals
                  19 = Unknown source of precipitation 
                  21 = Ground fog 
                  22 = Ice fog or freezing fog
		  
            WV** = Weather in the Vicinity where ** has one of the following 
	           values:
		   
		   01 = Fog, ice fog, or freezing fog (may include heavy fog)
		   03 = Thunder
		   07 = Ash, dust, sand, or other blowing obstruction
		   18 = Snow or ice crystals
		   20 = Rain or snow shower
		   
VALUE1     is the value on the first day of the month (missing = -9999).

MFLAG1     is the measurement flag for the first day of the month.  There are
           ten possible values:

           Blank = no measurement information applicable
           B     = precipitation total formed from two 12-hour totals
           D     = precipitation total formed from four six-hour totals
	   H     = represents highest or lowest hourly temperature (TMAX or TMIN) 
	           or the average of hourly values (TAVG)
	   K     = converted from knots 
	   L     = temperature appears to be lagged with respect to reported
	           hour of observation 
           O     = converted from oktas 
	   P     = identified as "missing presumed zero" in DSI 3200 and 3206
           T     = trace of precipitation, snowfall, or snow depth
	   W     = converted from 16-point WBAN code (for wind direction)

QFLAG1     is the quality flag for the first day of the month.  There are 
           fourteen possible values:

           Blank = did not fail any quality assurance check
           D     = failed duplicate check
           G     = failed gap check
           I     = failed internal consistency check
           K     = failed streak/frequent-value check
	   L     = failed check on length of multiday period 
           M     = failed megaconsistency check
           N     = failed naught check
           O     = failed climatological outlier check
           R     = failed lagged range check
           S     = failed spatial consistency check
           T     = failed temporal consistency check
           W     = temperature too warm for snow
           X     = failed bounds check
	   Z     = flagged as a result of an official Datzilla 
	           investigation

SFLAG1     is the source flag for the first day of the month.  There are 
           thirty possible values (including blank, upper and 
	   lower case letters):

           Blank = No source (i.e., data value missing)
           0     = U.S. Cooperative Summary of the Day (NCDC DSI-3200)
           6     = CDMP Cooperative Summary of the Day (NCDC DSI-3206)
           7     = U.S. Cooperative Summary of the Day -- Transmitted 
	           via WxCoder3 (NCDC DSI-3207)
           A     = U.S. Automated Surface Observing System (ASOS) 
                   real-time data (since January 1, 2006)
	   a     = Australian data from the Australian Bureau of Meteorology
           B     = U.S. ASOS data for October 2000-December 2005 (NCDC 
                   DSI-3211)
	   b     = Belarus update
	   C     = Environment Canada
	   D     = Short time delay US National Weather Service CF6 daily 
	           summaries provided by the High Plains Regional Climate
		   Center
	   E     = European Climate Assessment and Dataset (Klein Tank 
	           et al., 2002)	   
           F     = U.S. Fort data 
           G     = Official Global Climate Observing System (GCOS) or 
                   other government-supplied data
           H     = High Plains Regional Climate Center real-time data
           I     = International collection (non U.S. data received through
	           personal contacts)
           K     = U.S. Cooperative Summary of the Day data digitized from
	           paper observer forms (from 2011 to present)
           M     = Monthly METAR Extract (additional ASOS data)
	   m     = Data from the Mexican National Water Commission (Comision
	           National del Agua -- CONAGUA)
	   N     = Community Collaborative Rain, Hail,and Snow (CoCoRaHS)
	   Q     = Data from several African countries that had been 
	           "quarantined", that is, withheld from public release
		   until permission was granted from the respective 
	           meteorological services
           R     = NCEI Reference Network Database (Climate Reference Network
	           and Regional Climate Reference Network)
	   r     = All-Russian Research Institute of Hydrometeorological 
	           Information-World Data Center
           S     = Global Summary of the Day (NCDC DSI-9618)
                   NOTE: "S" values are derived from hourly synoptic reports
                   exchanged on the Global Telecommunications System (GTS).
                   Daily values derived in this fashion may differ significantly
                   from "true" daily data, particularly for precipitation
                   (i.e., use with caution).
	   s     = China Meteorological Administration/National Meteorological Information Center/
	           Climatic Data Center (http://cdc.cma.gov.cn)
           T     = SNOwpack TELemtry (SNOTEL) data obtained from the U.S. 
	           Department of Agriculture's Natural Resources Conservation Service
	   U     = Remote Automatic Weather Station (RAWS) data obtained
	           from the Western Regional Climate Center	   
	   u     = Ukraine update	   
	   W     = WBAN/ASOS Summary of the Day from NCDC's Integrated 
	           Surface Data (ISD).  
           X     = U.S. First-Order Summary of the Day (NCDC DSI-3210)
	   Z     = Datzilla official additions or replacements 
	   z     = Uzbekistan update
	   
	   When data are available for the same time from more than one source,
	   the highest priority source is chosen according to the following
	   priority order (from highest to lowest):
	   Z,R,D,0,6,C,X,W,K,7,F,B,M,m,r,E,z,u,b,s,a,G,Q,I,A,N,T,U,H,S
	   
	   
VALUE2     is the value on the second day of the month

MFLAG2     is the measurement flag for the second day of the month.

QFLAG2     is the quality flag for the second day of the month.

SFLAG2     is the source flag for the second day of the month.

... and so on through the 31st day of the month.  Note: If the month has less 
than 31 days, then the remaining variables are set to missing (e.g., for April, 
VALUE31 = -9999, MFLAG31 = blank, QFLAG31 = blank, SFLAG31 = blank).
"""
ghcnd_all_fwf_header = [
    "ID",
    "Year",
    "Month",
    "Element",
    "Value1",
    "MFlag1",
    "QFlag1",
    "SFlag1",
    "Value2",
    "MFlag2",
    "QFlag2",
    "SFlag2",
    "Value3",
    "MFlag3",
    "QFlag3",
    "SFlag3",
    "Value4",
    "MFlag4",
    "QFlag4",
    "SFlag4",
    "Value5",
    "MFlag5",
    "QFlag5",
    "SFlag5",
    "Value6",
    "MFlag6",
    "QFlag6",
    "SFlag6",
    "Value7",
    "MFlag7",
    "QFlag7",
    "SFlag7",
    "Value8",
    "MFlag8",
    "QFlag8",
    "SFlag8",
    "Value9",
    "MFlag9",
    "QFlag9",
    "SFlag9",
    "Value10",
    "MFlag10",
    "QFlag10",
    "SFlag10",
    "Value11",
    "MFlag11",
    "QFlag11",
    "SFlag11",
    "Value12",
    "MFlag12",
    "QFlag12",
    "SFlag12",
    "Value13",
    "MFlag13",
    "QFlag13",
    "SFlag13",
    "Value14",
    "MFlag14",
    "QFlag14",
    "SFlag14",
    "Value15",
    "MFlag15",
    "QFlag15",
    "SFlag15",
    "Value16",
    "MFlag16",
    "QFlag16",
    "SFlag16",
    "Value17",
    "MFlag17",
    "QFlag17",
    "SFlag17",
    "Value18",
    "MFlag18",
    "QFlag18",
    "SFlag18",
    "Value19",
    "MFlag19",
    "QFlag19",
    "SFlag19",
    "Value20",
    "MFlag20",
    "QFlag20",
    "SFlag20",
    "Value21",
    "MFlag21",
    "QFlag21",
    "SFlag21",
    "Value22",
    "MFlag22",
    "QFlag22",
    "SFlag22",
    "Value23",
    "MFlag23",
    "QFlag23",
    "SFlag23",
    "Value24",
    "MFlag24",
    "QFlag24",
    "SFlag24",
    "Value25",
    "MFlag25",
    "QFlag25",
    "SFlag25",
    "Value26",
    "MFlag26",
    "QFlag26",
    "SFlag26",
    "Value27",
    "MFlag27",
    "QFlag27",
    "SFlag27",
    "Value28",
    "MFlag28",
    "QFlag28",
    "SFlag28",
    "Value29",
    "MFlag29",
    "QFlag29",
    "SFlag29",
    "Value30",
    "MFlag30",
    "QFlag30",
    "SFlag30",
    "Value31",
    "MFlag31",
    "QFlag31",
    "SFlag31",
]

ghcnd_all_fwf_widths = [
    11,
    4,
    2,
    4,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
    5,
    1,
    1,
    1,
]

"""
ID         is the station identification code.  Note that the first two
           characters denote the FIPS  country code, the third character 
           is a network code that identifies the station numbering system 
           used, and the remaining eight characters contain the actual 
           station ID. 

           See "ghcnd-countries.txt" for a complete list of country codes.
	   See "ghcnd-states.txt" for a list of state/province/territory codes.

           The network code  has the following five values:

           0 = unspecified (station identified by up to eight 
	       alphanumeric characters)
	   1 = Community Collaborative Rain, Hail,and Snow (CoCoRaHS)
	       based identification number.  To ensure consistency with
	       with GHCN Daily, all numbers in the original CoCoRaHS IDs
	       have been left-filled to make them all four digits long. 
	       In addition, the characters "-" and "_" have been removed 
	       to ensure that the IDs do not exceed 11 characters when 
	       preceded by "US1". For example, the CoCoRaHS ID 
	       "AZ-MR-156" becomes "US1AZMR0156" in GHCN-Daily
           C = U.S. Cooperative Network identification number (last six 
               characters of the GHCN-Daily ID)
	   E = Identification number used in the ECA&D non-blended
	       dataset
	   M = World Meteorological Organization ID (last five
	       characters of the GHCN-Daily ID)
	   N = Identification number used in data supplied by a 
	       National Meteorological or Hydrological Center
	   R = U.S. Interagency Remote Automatic Weather Station (RAWS)
	       identifier
	   S = U.S. Natural Resources Conservation Service SNOwpack
	       TELemtry (SNOTEL) station identifier
           W = WBAN identification number (last five characters of the 
               GHCN-Daily ID)

LATITUDE   is latitude of the station (in decimal degrees).

LONGITUDE  is the longitude of the station (in decimal degrees).

ELEVATION  is the elevation of the station (in meters, missing = -999.9).


STATE      is the U.S. postal code for the state (for U.S. stations only).

NAME       is the name of the station.

GSN FLAG   is a flag that indicates whether the station is part of the GCOS
           Surface Network (GSN). The flag is assigned by cross-referencing 
           the number in the WMOID field with the official list of GSN 
           stations. There are two possible values:

           Blank = non-GSN station or WMO Station number not available
           GSN   = GSN station 

HCN/      is a flag that indicates whether the station is part of the U.S.
CRN FLAG  Historical Climatology Network (HCN) or U.S. Climate Refererence
          Network (CRN).  There are three possible values:

           Blank = Not a member of the U.S. Historical Climatology 
	           or U.S. Climate Reference Networks
           HCN   = U.S. Historical Climatology Network station
	   CRN   = U.S. Climate Reference Network or U.S. Regional Climate 
	           Network Station

WMO ID     is the World Meteorological Organization (WMO) number for the
           station.  If the station has no WMO number (or one has not yet 
	   been matched to this station), then the field is blank. 
"""

ghcnd_stations_fwf_header = [
    "ID",
    "Latitude",
    "Longitude",
    "Elevation",
    "State",
    "Name",
    "GSNFlag",
    "HCNCRNFlag",
    "WMOID",
]

gazetteer_place_header = [
    "USPS",
    "GEOID",
    "ANSICODE",
    "NAME",
    "TYPE",
    "LSAD",
    "FUNCSTAT",
    "ALAND",
    "AWATER",
    "ALAND_SQMI",
    "AWATER_SQMI",
    "INTPTLAT",
    "INTPTLONG",
]


def download_ftp_file(in_url, in_filename):
    with closing(request.urlopen(in_url)) as r:
        with open(in_filename, "wb") as f:
            shutil.copyfileobj(r, f)


def download_and_extract(in_url, in_filename, in_dir, is_zip=True):
    if not Path(in_filename).is_file():
        download_ftp_file(in_url, in_filename)

    if not Path(in_dir).is_dir():
        if is_zip:
            with ZipFile(in_filename) as zip_ref:
                zip_ref.extractall(f"./{in_dir}")
        else:
            tar_fp = tarfile.open(in_filename)
            tar_fp.extractall(f"./{in_dir}/")
            tar_fp.close()


def place_on_bad_line(i_bl):
    bll = len(i_bl)
    nps = bll - 13
    return i_bl[:3] + [" ".join(i_bl[3 : 4 + nps])] + i_bl[4 + nps :]


def gazetteer_to_parquet(in_filename, is_zips=True):
    if Path(f"data/{in_filename}.parquet").is_file():
        return
    download_and_extract(
        f"https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/{in_filename}.zip",
        f"gazetteer_data/{in_filename}.zip",
        f"gazetteer_data/{in_filename}/",
    )
    if is_zips:
        r_df = pd.read_csv(
            f"gazetteer_data/{in_filename}/{in_filename}.txt", delim_whitespace=True
        )
    else:
        r_df = pd.read_csv(
            f"gazetteer_data/{in_filename}/{in_filename}.txt",
            delim_whitespace=True,
            names=gazetteer_place_header,
            header=0,
            skiprows=0,
            on_bad_lines=place_on_bad_line,
            engine="python",
        )
    r_df.drop(
        axis=1, labels=["ALAND", "AWATER", "ALAND_SQMI", "AWATER_SQMI"], inplace=True
    )
    r_df.to_parquet(f"db/{in_filename}.parquet", index=True)


def data_file_exist(in_filename, p_dir="data"):
    file_path = Path(p_dir) / in_filename
    return file_path.exists() and file_path.is_file()


def main():
    Path("data").mkdir(exist_ok=True, parents=True)
    Path("db").mkdir(exist_ok=True, parents=True)
    Path("gazetteer_data").mkdir(exist_ok=True, parents=True)

    print("Downloading All Temperature Data from NOAA")
    download_and_extract(
        "ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz",
        "ghcnd_all.tar.gz",
        "ghcnd_all",
        is_zip=False,
    )

    print("Transforming into usable Temperature DB")
    if not data_file_exist("ghcnd_all.csv"):
        read_files = list(Path("ghcnd_all").rglob("*.dly"))
        shown_header = False
        for f_l in tqdm(read_files):
            fwf_check = pd.read_fwf(f_l, widths=ghcnd_all_fwf_widths, header=None)
            if not shown_header:
                fwf_check.to_csv(
                    "data/ghcnd_all.csv",
                    index=False,
                    mode="a+",
                    sep=",",
                    encoding="utf-8-sig",
                    header=ghcnd_all_fwf_header,
                )
                shown_header = True
            else:
                fwf_check.to_csv(
                    "data/ghcnd_all.csv",
                    index=False,
                    mode="a+",
                    sep=",",
                    encoding="utf-8-sig",
                    header=None,
                )

    print("Building Weather Station DB")
    if not data_file_exist("ghcnd-stations.txt"):
        ftp_url = "ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
        download_ftp_file(ftp_url, "data/ghcnd-stations.txt")

    if not data_file_exist("station_data.csv"):
        fwf_check = pd.read_fwf(
            "data/ghcnd-stations.txt",
            widths=[11, 9, 10, 7, 3, 31, 4, 4, 6],
            header=None,
        )
        fwf_check.to_csv(
            "data/station_data.csv",
            index=False,
            mode="w",
            sep=",",
            encoding="utf-8-sig",
            header=ghcnd_stations_fwf_header,
        )

    print("Building Monthly Average Temperature DB")
    if not data_file_exist("month_avg_data.csv"):
        csv_line_headers = None
        with open("data/month_avg_data.csv", "w") as wfp:
            wfp.write("ID,Year,Month,Average\n")
            f_size = os.path.getsize("data/ghcnd_all.csv")
            n_lines_approx = int(np.ceil(f_size / 215.0))
            with open("data/ghcnd_all.csv", "r") as fp:
                for line in tqdm(fp, total=n_lines_approx):
                    line = line.strip()
                    line_arr = line.split(",")
                    if csv_line_headers is None:
                        line_arr[0] = "ID"
                        csv_line_headers = line_arr.copy()
                    else:
                        line_dict = dict(zip(csv_line_headers, line_arr))
                        if line_dict["Element"] != "TMAX":
                            continue
                        month_temp_num = 0
                        month_temp_total = 0.0
                        for k, v in line_dict.items():
                            if "Value" in k:
                                cur_val = float(v)
                                if cur_val < -1000:
                                    continue
                                month_temp_num += 1
                                month_temp_total += cur_val
                        month_avg = month_temp_total / month_temp_num
                        wfp.write(
                            f"{line_dict['ID']},{line_dict['Year']},{line_dict['Month']},{month_avg}\n"
                        )

    print("Building Location and Temperature DB")
    if not data_file_exist("loc_to_temp_db.parquet", "db"):
        m_avg_data = pd.read_csv("data/month_avg_data.csv")
        station_data = pd.read_csv("data/station_data.csv")

        y_avg_temp = (
            m_avg_data.groupby(by=["ID", "Year"]).mean().drop(labels=["Month"], axis=1)
        )
        y_avg_temp["Average"] /= 10.0

        both_inds = pd.DataFrame()
        both_inds["ID"] = y_avg_temp.index.get_level_values(0)
        both_inds["Year"] = y_avg_temp.index.get_level_values(1)

        s_data_w_year = pd.merge(station_data, both_inds, how="left", on=["ID"])
        s_data_w_year.set_index(["ID", "Year"], inplace=True)

        all_data = y_avg_temp.merge(s_data_w_year, how="left", on=["ID", "Year"])
        # all_data["F_Average"] = (all_data["Average"]*9/5) + 32
        all_data.to_parquet("db/loc_to_temp_db.parquet", index=True)

    print("Downloading and Parsing Gazetteer data")
    gazetteer_to_parquet("2021_Gaz_place_national", False)

    gazetteer_to_parquet("2021_Gaz_zcta_national")

    print("Building Final DB")
    db_path = Path("database.duckdb")
    db_path.unlink(missing_ok=True)
    con = ddb.connect(database="database.duckdb")
    con.execute(
        "CREATE TABLE loc_to_temp AS SELECT * FROM read_parquet('db/loc_to_temp_db.parquet')"
    )
    con.execute(
        "CREATE TABLE place_names AS SELECT * FROM read_parquet('db/2021_Gaz_place_national.parquet')"
    )
    con.execute(
        "CREATE TABLE place_zips AS SELECT * FROM read_parquet('db/2021_Gaz_zcta_national.parquet')"
    )
    
    print("Adding Macros")
    # Great Arc Distance
    # https://en.wikipedia.org/wiki/Great-circle_distance
    macros = {
        "dhav": ("th", "(sin(radians(th)/2)^2)"),
        "dlta": ("a", "b", "abs(b-a)"),
        "ahav": ("th", "2.0*asin(sqrt(th))"),
        "gad": ("long1", "lat1", "long2", "lat2", "degrees(ahav(dhav(dlta(lat1,lat2)) + (1 - dhav(dlta(lat1,lat2)) - dhav(lat1+lat2))*dhav(dlta(long1,long2))))")
    }
    for mname, mdef in macros.items():
        con.execute(f"DROP MACRO IF EXISTS {mname}")
        msig = f"{mname}({','.join(mdef[:-1])})"
        con.execute(f"CREATE MACRO {msig} AS {mdef[-1]}")
    
    print("All Done!")

    db_explain = (
        ("loc_to_temp", "Connects locations to a time series of yearly average temperatures"),
        ("place_names", "Connects names of places to latitude & longitude"),
        ("place_zips", "Connects zip codes to latitude & longitude"),
    )

    mnl = max(*tuple(len(xx[0]) for xx in db_explain))
    mll = max(*tuple(len(xx[1]) for xx in db_explain))
    print("============================================")
    print("All data can now be found in database.duckdb")
    print("Table", " " * (mnl - 1), "Comment")
    print("-" * (mnl + mll + 4))
    for db_name, db_comm in db_explain:
        print(db_name, " " * (mnl - len(db_name) + 4), db_comm)


if __name__ == "__main__":
    main()
