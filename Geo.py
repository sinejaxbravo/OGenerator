import geocoder


def location():
    user = geocoder.ip('me')
    city = user.city
    zip = user.postal
    print(zip)
    return zip


location()