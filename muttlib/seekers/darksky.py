"""This module extracts weather data from DarkSky API."""
import json

import pandas as pd
import requests

BASE_DARKSKY_API_URL = "https://api.darksky.net"


class DarkSkyWeatherFetcher(object):
    """Dark Sky API weather fetcher.

    Notes:
    - Units are present in the `units` attribute of the return value (check docs).

    Ref:
    https://darksky.net/dev/docs
    """

    def __init__(self, api_key):  # noqa
        self.api_key = api_key

    def _query_darksky_query(
        self,
        url,
        extraction_date=None,
        target_date=None,
        timeout=None,
        exclude=["currently", "minutely", "alerts", "daily"],
        extend="hourly",
        **params,
    ):
        """Perform API query."""
        if type(exclude) in [tuple, list]:
            exclude = ",".join(exclude)

        params.update(dict(exclude=exclude, extend=extend))
        request_params = {
            "params": params,
            "headers": {"Accept-Encoding": "gzip"},
            "timeout": timeout,
        }
        response = requests.get(url, **request_params)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError("Bad response")

        rv = json.loads(response.text)
        units = rv['flags']['units']
        rv = self._format_darksky_response(rv)
        rv["extraction_datetime"] = extraction_date
        rv["target_date"] = target_date
        rv["units"] = units
        return rv

    def query_forecast(self, **params):
        """Perform API query to get forecast data."""
        url = "{BASE_DARKSKY_API_URL}/forecast/{api_key}/{latitude},{longitude}".format(
            BASE_DARKSKY_API_URL=BASE_DARKSKY_API_URL, api_key=self.api_key, **params
        )
        rv = self._query_darksky_query(url, **params)
        return rv

    def query_observed(self, **params):
        """Perform API query to get observed data.

        Notes:
            The API expects a UTC timestamp for a give lat / long location and has a, difficult-to-overlook mechanism of interpreting that time in the request +
            transforming it into a response.

            It will then shift the request timestamp to the local (lat/long)
            datestamp, with a corrected timezone. Then it'll just keep the date part of
            this datestamp. The consequences of this is that the timestamp might be,
            for example, for 21/02/19 whilst the local date is set to 20/02 locally.
            After, it'll truncate any hours/minutes/seconds from the datestamp and
            calculate the weather for starting from this truncated datestamp, up
            until 24hrs later. This means he response is in timestamp format so it'll
            be in UTC timezone, but on the hours previously mentioned. This means that
            it returns the weather conditions for the 24 hours of the whole requested from local midnight of one day to the next.
            Thus, when requesting data from a GMT-4 timezone the first
            and last items will have times ranging from 04:00:00 to 03:00:00 of the
            following day.

            Remember the response might have missing data, as detailed in the api docs.

            For more info please see: https://darksky.net/dev/docs#time-machine-request
            that reads:
            `The daily data block will contain a single data point referring to the
            requested date.`
        """
        url = "{BASE_DARKSKY_API_URL}/forecast/{api_key}/{latitude},{longitude},{ts}".format(
            BASE_DARKSKY_API_URL=BASE_DARKSKY_API_URL,
            api_key=self.api_key,
            ts=int(params['target_date'].timestamp()),
            **params,
        )
        rv = self._query_darksky_query(url, **params)
        return rv

    def _format_darksky_response(self, data):
        """Convert raw darksky api response into a df suitable for processing."""
        fields = {
            ("apparentTemperature", "apparent_temperature"),
            ("cloudCover", "cloud_cover"),
            ("dewPoint", "dew_point"),
            ("humidity", "humidity_percentage"),
            ("icon", "icon"),
            ("precipIntensity", "precip_intensity"),
            ("precipProbability", "precip_probability"),
            ("precipType", "precip_type"),
            ("pressure", "pressure"),
            ("summary", "summary"),
            ("temperature", "temperature"),
            ("time", "time"),
            ("uvIndex", "uv_index"),
            ("visibility", "visibility"),
            ("windBearing", "wind_bearing"),
            ("windGust", "wind_gust"),
            ("windSpeed", "wind_speed"),
        }
        rows = []
        for e in data["hourly"]["data"]:
            rows.append({new_name: e.get(old_name) for old_name, new_name in fields})
        df = pd.DataFrame(rows)
        df["weather_datetime"] = pd.to_datetime(df["time"], unit="s")
        df.drop(columns=["time"])
        return df
