from kivy.app import App
from kivy.uix.widget import Widget
import controller

# https://kivy.org/doc/stable/gettingstarted/installation.html#install-pip

class ViewApp(App):
    def build(self):
        return controller()



if __name__ == '__main__':
    View().run()