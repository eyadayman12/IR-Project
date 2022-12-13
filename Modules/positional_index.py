def positional_index(tokenized_docs):
    pos_index = {}
    file_number = 0
    for tokenized_doc in tokenized_docs:
        for position, word in enumerate(tokenized_doc):
            if word in pos_index:
                pos_index[word][0] = pos_index[word][0] + 1
                if file_number in pos_index[word][1]:
                    pos_index[word][1][file_number].append(position)

                else:
                    pos_index[word][1][file_number] = [position]
            else:
                pos_index[word] = []
                pos_index[word].append(1)
                pos_index[word].append({})     
                pos_index[word][1][file_number] = [position]
        
        file_number+=1
    return pos_index