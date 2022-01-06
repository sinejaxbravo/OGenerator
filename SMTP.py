import sys
import datetime
import smtplib

gmail_user = 'littlerascalsfall21@gmail.com'
gmail_password = 'Littlerascals1!'

try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)

    to_addr = sys.argv[1]
    from_addr = "Little Rascals <littlerascalsfall21@gmail.com>"
    msg_type = sys.argv[2]

    if int(msg_type) == 1:
        subject_of_message = "Form successfully submitted"
        date_and_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        message_text = "Congrats!\nYou submitted a form at %s\n\n-Littlerascals :)" % date_and_time
        message = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % (from_addr, to_addr, subject_of_message,
                                                                     date_and_time, message_text)
    elif int(msg_type) == 2:
        subject_of_message = "You have successfully created an account"
        date_and_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        message_text = "Account created at %s for email: %s \n\n-Littlerascals :)" % (date_and_time, to_addr)
        message = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % (from_addr, to_addr, subject_of_message,
                                                                     date_and_time, message_text)
    elif int(msg_type) == 3:
        subject_of_message = "You have been assigned a new poll"
        date_and_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        message_text = "A new poll assignment was created at %s\n\n-Littlerascals" % date_and_time
        message = "From: %s\nTo: %s\nSubject: %s\nDate: %s\n\n%s" % (from_addr, to_addr, subject_of_message,
                                                                     date_and_time, message_text)

    server.sendmail(from_addr, to_addr, message)
    server.quit()
except:
    print('Error sending email! Make sure that python is installed and on path!')
wrong...')