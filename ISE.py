import sys
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import spacy
import spacy_help_functions
from spanbert import SpanBERT
import requests
import re
import os
import openai
import time

class ISE:
    
    internal_relations = { 1: "per:schools_attended",
                        2: "per:employee_of",
                        3: "per:cities_of_residence",
                        4: "org:top_members/employees" }
    RELATIONS = {1: "Schools_Attended",
                 2: "Work_For",
                 3: "Live_In",
                 4: "Top_Member_Employees"}
    
    INTEREST_ENTITY = { 1: {"subj": {"PERSON"}, "obj":{"ORGANIZATION"}},
                        2: {"subj": {"PERSON"}, "obj":{"ORGANIZATION"}},
                        3: {"subj": {"PERSON"}, "obj":{"LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"}},
                        4: {"subj": {"ORGANIZATION"}, "obj": {"PERSON"}} }
                

    def retrieve_plain_text(self,url):
        response = requests.get(url)
        html_content = response.content
        #print(type(content))
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract plain text content from the webpage
        text = soup.get_text()

        # Print the extracted text
        #print(text)
        #Removing redundant newlines and some whitespace characters
        preprocessed_text = re.sub(u'\xa0', ' ', text) 
        preprocessed_text = re.sub('\t+', ' ', preprocessed_text) 
        preprocessed_text = re.sub('\n+', ' ', preprocessed_text) 
        preprocessed_text = re.sub(' +', ' ', preprocessed_text) 
        preprocessed_text = preprocessed_text.replace('\u200b', '')
        return preprocessed_text
    
    def candidate_entity_pairs(self,sentence,entity_of_interest,subj,obj):
        candidate_pairs = []
        entity_pairs = spacy_help_functions.create_entity_pairs(sentence, entity_of_interest)
        for entity_pair in entity_pairs:
            #only keep entity pairs which contain both subj and obj in entity of interest
            if entity_pair[1][1] in subj and entity_pair[2][1] in obj:
                candidate_pairs.append({"tokens": entity_pair[0], "subj": entity_pair[1], "obj": entity_pair[2]})
            elif entity_pair[1][1] in obj and entity_pair[2][1] in subj:
                candidate_pairs.append({"tokens": entity_pair[0], "subj": entity_pair[2], "obj": entity_pair[1]})
        return candidate_pairs
    
    def spanbert_predicted_relations(self,spanbert,candidate_pairs,X,internal_name,t):
        new_relations = []
        duplicate_relations = []
        relation_preds = spanbert.predict(candidate_pairs)
        for i in range(len(relation_preds)):
            cp = candidate_pairs[i]
            pred = relation_preds[i]
            #print(cp)
            #print((pred[0],pred[1]))

            if pred[0] == internal_name and pred[1] >= t:
                tup = tuple([cp['subj'][0],cp['obj'][0],pred[1]]) # Set X contains tuples: (sub,obj,confidence)
                key = cp['subj'][0] + " " + cp['obj'][0]
                if key in X.keys():
                    if pred[1] > X[key][2]:
                        X[key] = tup
                    else:
                        duplicate_relations.append(tup)
                else:
                    X[key] = tup
                    new_relations.append(tup)
        return new_relations,duplicate_relations
    
    def gpt3_predicted_relations(self,entity_of_interest,relation,sentence,subj,obj,X):
        new_relations = []
        duplicate_relations = []
        entity_pairs = spacy_help_functions.create_entity_pairs(sentence, entity_of_interest)
        should_process = False
        for entity_pair in entity_pairs:
            if entity_pair[1][1] in subj and entity_pair[2][1] in obj or entity_pair[1][1] in obj and entity_pair[2][1] in subj:
                should_process = True
        #print(should_process)
        if should_process:
            sentence_text = sentence.text
            #print(sentence_text)
            #prompt = "Given a paragraph, extract all instances of the following relationship types as possilbe ",sentence_text,
            prompt =  (
                f"Given a paragraph, extract all instances of the '{relation}' relationship types as possible, if nothing is found return no relation\n" 
                "Output format for one relation: ['SUBJECT ENTITY', 'RELATIONSHIP', 'OBJECT ENTITY'].\n"
                "If there're multiple satisfying relations, separate them by comma\n" 
                f"Paragraph: {sentence_text}\n" 
                "Example output for only one relation founded: ['SUBJECT ENTITY', 'RELATIONSHIP', 'OBJECT ENTITY']\n" 
                "Example output for multiple relations founded: ['SUBJECT ENTITY1', 'RELATIONSHIP', 'OBJECT ENTITY1'],['SUBJECT ENTITY2', 'RELATIONSHIP', 'OBJECT ENTITY2']\n"
            )
            #print("This is prompt  ", prompt)
            #f"Sample Output: [{relation}]"
            #print(sentence)
            res = self.get_openai_completion(prompt, model = 'text-davinci-003', max_tokens = 100, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty =0)
            #print(res)
            pattern = r"\[.*?\]"#r"\['(.*?)', '(.*?)', '(.*?)'\]" #rf'.*{relation}.*\((.*),\s*(.*)\)'
            match = re.findall(pattern, res)
            #print(match)
            if match:
                for item in match:
                    pat = r"\['(.*?)', '(.*?)', '(.*?)'\]"
                    this_match = re.search(pat,item)
                    subject, relationship, object = this_match.groups()
                #print("gpt results: " ,subject,object)
                    tup = tuple([subject,object])
                    key = subject + " " + object
                    if key in X.keys():
                        duplicate_relations.append(tup)
                    else:
                        X[key] = tup
                        new_relations.append(tup)    
            # else:
            #     print("No match found")
        time.sleep(1.5)
        return new_relations,duplicate_relations

    def get_openai_completion(self,prompt, model = 'text-davinci-003', max_tokens = 100, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty =0):
        response = openai.Completion.create(
            model = model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        response_text = response['choices'][0]['text']
        return response_text 

    def main(self):
        print(sys.argv)
        # api_key = sys.argv[1]
        # eng_id = sys.argv[2]
        # threshold = float(sys.argv[3])
        # query = sys.argv[4]
            # Retrieve the command line arguments
        args = sys.argv[1:]

        # Check if a language model has been specified
        if '-spanbert' in args:
            method = 'spanbert'
            args.remove('-spanbert')
        elif '-gpt3' in args:
            method = 'gpt3'
            args.remove('-gpt3')
        else:
            print('Please specify a language model to use with either -spanbert or -gpt3')
            return

        # Retrieve the remaining arguments
        api_key = args[0]
        eng_id = args[1]
        openai_key = args[2]
        r = int(args[3])
        t = float(args[4])
        q = args[5]
        k = int(args[6]) 
        # Initialize X, the set of extracted tuples, as the empty set.
        X = {}
        queriedTuple = set()
        queriedTuple.add(q)
       
        # get interested entity based on input relation
        subj = self.INTEREST_ENTITY[r]['subj']
        obj = self.INTEREST_ENTITY[r]['obj']
        # get the internal name of the desired predict relation
        internal_name = self.internal_relations[r]
        relation = self.RELATIONS[r]
        openai.api_key = openai_key
        visitedUrl = set()
        iteration = 0
        print("----------------------------------------------------------------------------")
        print("Parameters:")
        print("Client Key        = ", api_key)
        print("Engine Key        = ", eng_id)
        print("OpenAI key        = ",openai_key)
        print("Method            = ",method)
        print("Relation          = ", self.RELATIONS[r])
        print("Threshold:", t)
        print("Query:", q)
        print("# of Tuples:", k)
        print("Loading necessary libraries; This should take a minute or so ...")
        # Load spacy model
        spacy_model = spacy.load("en_core_web_lg")
        # Load pre-trained SpanBERT model
        if method == "spanbert":
            spanbert = SpanBERT("./pretrained_spanbert")
        #print(type(spanbert))
        while(True):
            print("==================== Iteration:", iteration, "- Query:", q, "====================")
            print()
            print()
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=q,cx=eng_id).execute()
            #print(res)
            if res == None:
                break
            if "items" not in res:
                break
            items = res["items"]
            URLs = []
            for item in items:
                URLs.append(item['link'])
            #print(URLs)

            for i,url in enumerate(URLs):
                if url in visitedUrl:
                    print("URL (", i + 1, "/ 10): This link has already been visited, skip!")
                    continue
                print("URL (", i + 1, "/ 10):", url) 
                visitedUrl.add(url)
                # Retrieve the content of the webpage corresponding to the url
                print("     Fetching text from url ...")
                text = self.retrieve_plain_text(url)
                #print(text)
                length = len(text)
                if length >= 10000:
                    print("     Trimming webpage content from", length, "to 10000 characters")
                    text = text[0:10000]
                print("         Webpage Length (num characters):", len(text))
                
                print("     Annotating the webpage using spacy...")
                # apply spacy model to extract entities
                doc = spacy_model(text)
                num = 0
                for sentence in doc.sents:
                    num += 1
                    #print(type(sentence))
                print("     Extracted", num, "sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
                #get entities that we want spacy to extract 
                entity_of_interest = []
                for subj_ent in subj:
                    entity_of_interest.append(subj_ent)
                for obj_ent in obj:
                    entity_of_interest.append(obj_ent)
                relations_this_web = 0
                for idx,sentence in enumerate(doc.sents):
                    candidate_pairs = self.candidate_entity_pairs(sentence,entity_of_interest,subj,obj)
                    if len(candidate_pairs) == 0:
                        continue
                    #print(candidate_pairs)
                    if method == "spanbert":
                        new_relations,duplicate_relations = self.spanbert_predicted_relations(spanbert,candidate_pairs,X,internal_name,t)
                    elif method == "gpt3":
                        new_relations,duplicate_relations = self.gpt3_predicted_relations(entity_of_interest,relation,sentence,subj,obj,X)
                        #print(new_relations)
                        #print(duplicate_relations)
                    relations_this_web += len(new_relations)
                    #print(new_relations)
                    #print(duplicate_relations)
                    if len(new_relations) != 0 or len(duplicate_relations) != 0:
                        print()
                        print("                     === Extracted Relation ===")
                        print("                     Input tokens:   ", [token.text for token in sentence])
                        if len(new_relations) != 0:
                            for tup in new_relations:
                                if method == "spanbert":
                                    print("             Output Confidence: ", tup[2], " ; Subject: ", tup[0], " ; Object: ", tup[1], " ;")
                                elif method == "gpt3":
                                    print("             Subject: ", tup[0], " ; Object: ", tup[1], " ;")
                            print("                 Adding to set of extracted relations")
                        if len(duplicate_relations) != 0 and method == "spanbert":
                            for tup in duplicate_relations:
                                print("             Output Confidence: ", tup[2], " ; Subject: ", tup[0], " ; Object: ", tup[1], " ;")
                                print("             Duplicate with lower confidence than existing record. Ignoring this.")
                        print("          ==========")
                        print()
                    if (idx + 1) % 5 == 0:
                        print("     Processed", idx + 1, "/", num, "sentences")

                print("             Relations extracted form this website: ", relations_this_web)
                print()
                print()

            # sort X by confidence
            if method == "spanbert":
                X = dict(sorted(X.items(), key=lambda x: x[1][2], reverse = True))
            
            if len(X) >= k:
                break
            
            y_exists = False
            # select the relation that has not been queried before with the highest confidence as a new query
            for key, val in X.items():
                if key not in queriedTuple:
                    print("new relation to be appended to the query:", key)
                    queriedTuple.add(key)
                    y_exists = True
                    q = q + " " + key
                    break

            # stop if such relation y does not exists
            if y_exists == False:
                break
            iteration += 1

        print("================== ALL RELATIONS for ",internal_name, " ( ", len(X), " ) =================")
        for key,val in X.items():
            if method == "spanbert":
                print("CONFIDENCE:", val[2], "\t", "  |SUBJECT:", val[0], "\t", "  |OBJECT:", val[1])
            elif method == "gpt3":
                print("SUBJECT:", val[0], "\t", "  |OBJECT:", val[1])
        print(print("Total # of iterations =", iteration + 1))
if __name__ == '__main__':
    ise = ISE()
    ise.main()