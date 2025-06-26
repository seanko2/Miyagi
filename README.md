# Miyagi

## Mission
My mission for this project is to give job seekers FREE access to practice on their behaviorial interview skills. Most interview prep tools that are out there 
are usually paid and can be a big financial obstacle for some. I want to remove that obstacle for students like me that need practice with interviewing but did not know an easy way to find it.


## Structure
1. Data Collection
2. Data Embedding
3. RAG System
4. Voice mode
5. Chat Model
6. Limitations


### Data Collection
I knew there are information scattered online about how to nail behaviorial interview and best practices like the STAR Method. However, I wanted Miyagi to specialize in
this field and consolidate all those information in one through Youtube, websites, and also Ebooks. I want to preface that maybe the strong LLM models that are out there already have
trained on those documents. However, I am using a smaller model (more on that later) and did not want to risk any hallucinations, which motivated me to build RAG system 
(Also more on this later). 

Method:
Gather a lot of urls (manually) of websites, youtube videos, Ebooks and scrape the text from them. Then store it into a .txt file in AWS. I hand picked these websites
which made this process SUPER tedious but the data is ensured be to of high quality. A lot of websites blocked crawler made this process more manual. 

Packages I used:
BeautifulSoup (website specific. Documentation: https://beautiful-soup-4.readthedocs.io/en/latest/)\n
requests
re
PYPDF2 (Ebook specific. Documentation: https://pypdf2.readthedocs.io/en/3.x/)
Youtube_transcript_api (Easy to use! Documentation: https://pypi.org/project/youtube-transcript-api/)
boto3 (AWS S3 storage specific)
