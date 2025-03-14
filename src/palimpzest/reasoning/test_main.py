import json

from palimpzest.reasoning.orchestrator import Orchestrator

question = """
I have datasources and need to extract (may include simple format transformations) the following fields from the datasource:
- sender: The email address of the sender
- subject: The subject of the email

Below are some examples of datasources and the expected results annotated from human experts:

<example-0>
<datasource>
"Message-ID: <1390685.1075853083264.JavaMail.evans@thyme>\nDate: Mon, 17 Sep 2001 07:56:52 -0700 (PDT)\nFrom: steven.january@enron.com\nTo: shelley.corman@enron.com, lynn.blair@enron.com, rick.dietz@enron.com, \n\tbradley.holmes@enron.com, donna.scott@enron.com, \n\tmike.bryant@enron.com, sharon.brown@enron.com, \n\tdarrell.schoolcraft@enron.com, gary.spraggins@enron.com, \n\tdale.ratliff@enron.com, ricki.winters@enron.com\nSubject: VACATION\nMime-Version: 1.0\nContent-Type: text/plain; charset=us-ascii\nContent-Transfer-Encoding: 7bit\nX-From: January, Steven </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SJANUARY>\nX-To: Corman, Shelley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Scorman>, Blair, Lynn </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Lblair>, Dietz, Rick </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rdietz>, Holmes, Bradley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Bholmes>, Scott, Donna </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dscott1>, Bryant, Mike </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Mbryant>, Brown, Sharon </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Sbrown1>, Schoolcraft, Darrell </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dschool>, Spraggins, Gary </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Gspragg>, Ratliff, Dale </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dratlif>, Winters, Ricki </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rwinter>\nX-cc: \nX-bcc: \nX-Folder: \\LBLAIR (Non-Privileged)\\Blair, Lynn\\Vacations 2001\nX-Origin: Blair-L\nX-FileName: LBLAIR (Non-Privileged).pst\n\nI plan on taking vacation October 4, 5, 8,9 10, 11, and 12. This will finish my vacation for the year. thanks. sj",
</datasource>
The experts give me the following annotations:
<annotated_fields>
- sender: "steven.january"
- subject: "Vacation"
</annotated_fields>
</example-0>

<example-1>
<datasource>
"Message-ID: <760490.1075853083334.JavaMail.evans@thyme>
Date: Fri, 6 Jul 2001 11:17:00 -0700 (PDT)
From: sheila.nacey@enron.com
To: lynn.blair@enron.com, mike.bryant@enron.com, shelley.corman@enron.com, 
	rick.dietz@enron.com, bradley.holmes@enron.com, 
	steven.january@enron.com, donna.scott@enron.com, 
	gina.taylor@enron.com
Subject: Vacation plans
Cc: ricki.winters@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
Bcc: ricki.winters@enron.com
X-From: Nacey, Sheila </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SNACEY>
X-To: Blair, Lynn </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Lblair>, Bryant, Mike </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Mbryant>, Corman, Shelley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Scorman>, Dietz, Rick </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rdietz>, Holmes, Bradley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Bholmes>, January, Steven </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Sjanuary>, Scott, Donna </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dscott1>, Taylor, Gina </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Gtaylor10>
X-cc: Winters, Ricki </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Rwinter>
X-bcc: 
X-Folder: \LBLAIR (Non-Privileged)\Blair, Lynn\Vacations 2001
X-Origin: Blair-L
X-FileName: LBLAIR (Non-Privileged).pst

I will be taking Monday, 7/16, off as a day of vacation.  Also, I have scheduled 8/24 thru 31 and 10/29 thru 11/5.  Almost afraid to go anywhere!  Sheila"
</datasource>
The experts give me the following annotations:
<annotated_fields>
- sender: "sheila.nacey"
- subject: "Vacation Plans"
</annotated_fields>
</example-1>

<question>
Please help me to write a prompt to extract the fields from the datasource to match users' expectations. 
- I have hundreds of emails with the similar structure and I need to extract the fields from each email, so please make the prompt general enough to work for any email.
- The returned answer is just the prompt itself and don't add any other text:
 - use {datasource} to refer to the datasource as I'll replace it with the actual datasource.
 - You already know the fields to extract, so don't add any other fields or use placeholder to the prompt.
 - The returned prompt will be used directly after fullfilling the {datasource} placeholder.
 - The prompt should be concise and to the point.
 - The prompt should ask for JSON output format.
</question>
"""

if __name__ == "__main__":
    agent = Orchestrator()
    response = agent.get_response(question)
    print(json.dumps(response, indent=4))
