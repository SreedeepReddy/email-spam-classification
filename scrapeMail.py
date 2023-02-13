import imaplib
import email

# Connect to Gmail using IMAP
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login("sreedeepreddyk@gmail.com", "password")

# Select the inbox folder
mail.select("inbox")

# Search for messages that match certain criteria
status, messages = mail.search(None, 'ALL')

# Extract information from the messages
for message in messages[0].split(b' '):
    status, msg = mail.fetch(message, "(RFC822)")
    for response in msg:
        if isinstance(response, tuple):
            msg = email.message_from_bytes(response[1])
            subject = msg["subject"]
            from_email = msg["from"]
            print(f"Subject: {subject}\nFrom: {from_email}\n")

# Close the connection
mail.close()
mail.logout()
