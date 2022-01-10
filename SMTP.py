import sys
import datetime
import smtplib

gmail_user = 'littlerascalsfall21@gmail.com'
gmail_password = 'Littlerascals1!'


class SMTP:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)

        # TODO MAKE AN ADDITIONAL EMAIL AND DONT USE THIS ONE.
        to_addr = ""
        from_addr = "Little Rascals <littlerascalsfall21@gmail.com>"

        date_and_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        subject_of_message = "change message when calling"
        message_text = "Congrats!\nYou signed up at %s\n\n-Jack :)" % date_and_time


        # TODO KEEP THIS THE SAME, ONLY CHANGE TO_ADDR, SUBJECT_OF_MSG AND MSG_TEXT
        message = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % (from_addr, to_addr, subject_of_message,
                                                                     date_and_time, message_text)

        def send_email(self):
            try:
                self.server.sendmail(self.from_addr, self.to_addr, self.message)
                self.server.quit()
            except:
                print('Unable to send message!!!')
