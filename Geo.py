import geocoder


def location():
    user = geocoder.ip('me')
    city = user.city
    zip = user.postal
    state = user.state
    current_location = (city, state, zip)
    return current_location

# location()