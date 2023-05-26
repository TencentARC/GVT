from .base_dataset import BaseDataset
import numpy as np

class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split

        names = ["vqav2_rest_val"]
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text_out"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        score_arr = np.array(scores)
        prob = score_arr / score_arr.sum()
        answer = np.random.choice(answers, p=prob)

        return {
            "image": image_tensor,
            "text_in": text,
            "text_out": answer,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }