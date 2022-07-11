import os
import html
import re
import sys
import json

from flask import Flask, redirect, url_for, request, render_template, send_from_directory, abort


import duckdb as ddb
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

con = ddb.connect(database='./database.duckdb', read_only=True)
app = Flask(
    __name__,
    static_folder="./static",
    template_folder="./templates"
)

si = html.escape

def render_df(in_df):
    return render_template(
        'tables.jinja2',
        tables=[in_df.to_html(classes='data')],
        titles=in_df.columns.values
    )


def is_float(in_str):
    try:
        float(in_str)
        return True
    except ValueError:
        return False

def is_int(in_str):
    try:
        int(in_str)
        return True
    except ValueError:
        return False

@app.errorhandler(404)
def not_found(e):
  return render_template("404.jinja2")

@app.route('/everywhere', methods=['GET'])
def everywhere():
    is_far = bool(si(request.args.get("use_f", default="")))
    t_expr = "((Average * 1.8) + 32)" if is_far else "Average"
    rv_data = con.execute(f"SELECT Year,AVG({t_expr}) AS T_Average FROM loc_to_temp GROUP BY Year").df().to_dict(orient='list')

    return render_template(
        "plot.jinja2",
        inputted="Everywhere",
        stations = "",
        x_pts = rv_data["Year"],
        y_pts = rv_data["T_Average"],
        is_f = is_far
    )

@app.route('/loc', methods=['GET'])
def get_data():
    i_llat = si(request.args.get("lat", default=""))
    i_llong = si(request.args.get("long", default=""))

    lcity = si(request.args.get("city", default=""))
    lst = si(request.args.get("state", default=""))

    lzip = si(request.args.get("zip", default=""))
    inputted = " ".join(map(str,[i_llat, i_llong, lcity, lst, lzip]))

    llat = -99.
    llong = -99.
    try:
        if bool(i_llat) and bool(i_llong) and is_float(i_llat) and is_float(i_llong):
            llat = i_llat
            llong = i_llong
            inputted = f"{i_llat}, {i_llong}"
        elif bool(lzip) and is_int(lzip):
            inputted = lzip
            con.execute("""
            SELECT INTPTLAT,INTPTLONG
            FROM place_zips
            WHERE GEOID = ?
            """, [lzip]
            )
            rv = con.fetchone()
            if rv is None:
                raise RuntimeError
            llat, llong = rv
        elif bool(lcity) and bool(lst):
            inputted = f"{lcity}, {lst}"
            con.execute("""
            SELECT INTPTLAT,INTPTLONG
            FROM place_names
            WHERE USPS = ? AND NAME ILIKE ?
            """, [lst, lcity]
            )
            rv = con.fetchone()
            if rv is None:
                raise RuntimeError
            llat, llong = rv
        else:
            raise RuntimeError

        is_far = bool(si(request.args.get("use_f", default="")))
        t_expr = "((Average * 1.8) + 32)" if is_far else "Average"
        con.execute(f"""
        SELECT ID,Year,{t_expr} AS Average,Name,Longitude,Latitude,gad(Longitude, Latitude, ?, ?)*69 AS Dist
        FROM loc_to_temp
        WHERE Dist < 35
        """, [llong, llat]
        ) 
        # the conversions aren't exact, but a roughly 
        # 35 mile radius seems right based on NOAA queries
    except RuntimeError:
            return render_template(
                'error.jinja2',
                msg=f"Bad Input or None Found! {inputted}"
            )
    
    rv_data = con.df()
    rv_temp = con.execute("SELECT Year,AVG(Average) AS avg FROM 'rv_data' GROUP BY Year ORDER BY Year").df().to_dict(orient='list')
    rv_stations = con.execute(
        "SELECT ID,first(Name) AS name,first(longitude) AS long,"
        "first(latitude) AS lat,first(Dist) AS dist "
        "FROM 'rv_data' GROUP BY ID Order BY dist"
    ).df().to_dict(orient='list')

    rv_stations["ID"].append("")
    rv_stations["name"].append("Resolved Location")
    rv_stations["long"].append(llong)
    rv_stations["lat"].append(llat)
    rv_stations["dist"].append(0.)

    return render_template(
        "plot.jinja2",
        inputted = inputted,
        stations = rv_stations,
        x_pts = rv_temp["Year"],
        y_pts = rv_temp["avg"],
        is_f = is_far
    )


@app.route('/plot_test')
def plot_test():
    return render_template("plot_test.jinja2")

@app.route('/')
def index_page():
    return render_template('index.jinja2')

@app.route('/<path:filename>')
def protected(filename):
	abort(404)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=10420)
