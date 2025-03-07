import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    return Nominatim, RateLimiter, mo, pd


@app.cell
def _(pd):
    df = pd.read_csv('Datasets\DKHousingPricesSample100k.csv')
    df.head(10)
    return (df,)


@app.cell
def _(Nominatim, RateLimiter, df, pd):
    df_subset = df.head(10)

    geolocator = Nominatim(user_agent="GeoSearcher")

    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    def get_lat_lon(row):
        try:
            address = row['address'].split(",")[0]
            zip_code = row['zip_code']
            city =  row['city'].split(" ")[0]
            full_address = f"{address}, {zip_code} {city}, Danmark"
        
            location = geocode(full_address)
        
            if location:
                return pd.Series([location.latitude, location.longitude])
    
        except Exception as es:
        
            print(f"{es}: {full_address}")
            return pd.Series([None, None])
    return df_subset, geocode, geolocator, get_lat_lon


@app.cell
def _(df_subset, get_lat_lon):
    df_subset[['latitude', 'longitude']]= df_subset.apply(get_lat_lon, axis=1)
    return


@app.cell
def _(df_subset):
    df_subset
    return


if __name__ == "__main__":
    app.run()
