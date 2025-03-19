import http.client, os, urllib

from agents.agent import Agent
from agents.deals import Opportunity

DO_TEXT = False
DO_PUSH = True

class MessagingAgent(Agent):

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        """
        Set up this object to either do push notifications via Pushover,
        or SMS via Twilio, whichever is specified in the constants.
        """
        super().__init__()
        self.initialize()

        if DO_TEXT:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'your-sid-if-not-using-env')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'your-auth-if-not-using-env')
            self.me_from = os.getenv('TWILIO_FROM', 'your-phone-number-if-not-using-env')
            self.me_to = os.getenv('MY_PHONE_NUMBER', 'your-phone-number-if-not-using-env')
            self.log("Connected to Twilio")
        if DO_PUSH:
            self.pushover_user = os.getenv('PUSHOVER_USER')
            self.pushover_token = os.getenv('PUSHOVER_TOKEN')
            self.log("Conected to Pushover")

        self.ready()


    def message(self, text):
        """
        Send an SMS message using the Twilio API
        """
        self.log("Sending text message")
        message = self.client.messages.create(
          from_=self.me_from,
          body=text,
          to=self.me_to
        )


    def push(self, text):
        """
        Send a Push Notification using the Pushover API
        """
        self.log("Sending push notification")

        payload = {
            "token": self.pushover_token,
            "user": self.pushover_user,
            "message": text,
            "sound": "cashregister"
        }

        conn = http.client.HTTPSConnection("api.pushover.net:443")

        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(payload),
            {"Content-type": "application/x-www-form-urlencoded"}
        )

        response = conn.getresponse()

        self.log(f"Response: {response.read()}")


    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        self.log("Constructing message")
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:10]+'... '
        text += opportunity.deal.url

        if DO_TEXT:
            self.message(text)
        if DO_PUSH:
            self.push(text)

        self.log("Done")

