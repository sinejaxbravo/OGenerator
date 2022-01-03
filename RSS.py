import tracemalloc

import Geo
import python_weather
import asyncio


async def extract_weather():
    client = python_weather.Client(format=python_weather.IMPERIAL)
    try:
        location = Geo.location()
        weather = await client.find(f"{location[0]} {location[1]}")
        print(weather.current.temperature)
    except:
        print("please enter your zip code")


tracemalloc.start()
asyncio.run(extract_weather())
