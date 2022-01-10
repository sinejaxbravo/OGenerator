import Geo
import python_weather



async def extract_weather():
    client = python_weather.Client(format=python_weather.IMPERIAL)
    try:
        location = Geo.location()
        weather = await client.find(f"{location[0]} {location[1]}")
        all_weather = weather.forecasts
        temperature = weather.current.temperature
        precipitation = weather.forecasts[0].precip
        overcast = weather.forecasts[0].sky_text
        print(f"Overhead: {overcast}")
        print(f"Temp: {temperature}")
        print(f"Precipitation: {precipitation}")
        await client.close()

        return temperature, precipitation, overcast
    except:
        print("Something went wrong gathering your weather information")


class Weather:
    pass
