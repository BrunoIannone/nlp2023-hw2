from typing import List
import json


class Vocabulary():
    """Vocabulary class
    """

    def __init__(self, sentences: List[List[str]], labels: List[List[str]], save_vocab: bool = False):
        """Init function for the vocabulary class

        Args:
            sentences (List[List[str]]): List of list of sentence tokens
            labels (List[List[str]]): List of list of sentences token labels
            save_vocab (bool, optional): True if the vocabulary needs to be saved (as .txt). Defaults to False.



        """
        token_vocabulary = self.build_tokens_vocabulary(sentences)

        self.word_to_idx = token_vocabulary["word_to_idx"]
        self.idx_to_word = token_vocabulary["idx_to_word"]

        labels_vocabulary = self.build_labels_vocabulary(labels)

        self.labels_to_idx = labels_vocabulary["labels_to_idx"]
        self.idx_to_labels = labels_vocabulary["idx_to_labels"]
        if save_vocab:
            with open("word_to_idx.txt", "a") as fp:
                json.dump(self.word_to_idx, fp)
                fp.close()
            with open("idx_to_word.txt", "a") as fp:
                json.dump(self.idx_to_word, fp)
                fp.close()
            with open("labels_to_idx.txt", "a") as fp:
                json.dump(self.labels_to_idx, fp)
                fp.close()

            with open("idx_to_labels.txt", "a") as fp:
                json.dump(self.idx_to_labels, fp)
                fp.close()

    def build_tokens_vocabulary(self, sentences: List[List[str]]):
        """Create two different vocabularies from sentence tokens for word to vocabulary index and viceversa
        N.B. Padding will have index 0 while unknown words index 1.
        Args:
            sentences (List[List[str]]): List of list of sentence tokens

        Returns:
            dict: {"word_to_idx": word_to_idx (dict), "idx_to_word": idx_to_word (dict)} 
        """
        word_to_idx = {}
        idx_to_word = {}
        special_characters = "!@#$%^&*()-+?_=,<>/"
        idx = 0

        word_to_idx["<pad>"] = idx
        idx_to_word[idx] = "<pad>"
        idx += 1

        word_to_idx["<unk>"] = idx
        idx_to_word[idx] = "<unk>"
        idx += 1

        for sentence_list in sentences:
            for token in sentence_list:

                if token.lower() not in word_to_idx and not any(character in special_characters for character in token.lower()):
                    word_to_idx[token.lower()] = idx
                    idx_to_word[idx] = token.lower()
                    idx += 1

        return {
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word
        }

    def build_labels_vocabulary(self, sentences_labels: List[List[str]]):
        """Create two different vocabularies from sentence labels for label to vocabulary index and viceversa.
        N.B.: padding will have the bigger index

        Args:
            sentences_labels (List[List[str]]): List of list of sentences token labels

        Returns:
            dict: {"labels_to_idx": labels_to_idx (dict), "idx_to_labels": idx_to_labels (dict)} 
        """
        labels_to_idx = {}
        idx_to_labels = {}
        idx = 0
        for label_list in sentences_labels:
            for label in label_list:
                if label not in labels_to_idx:
                    labels_to_idx[label] = idx
                    idx_to_labels[idx] = label
                    idx += 1

        labels_to_idx['<pad>'] = idx
        idx_to_labels[idx] = '<pad>'

        return {
            "labels_to_idx": labels_to_idx,
            "idx_to_labels": idx_to_labels
        }
