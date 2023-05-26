# GVTBench

## Object Counting
This task asks the model to predict the number of a certain object appeared in the image. 
We construct this task based on MS-COCO and Visual Commonsense Reasoning (VCR) datasets.

The annotation files are

[OC@COCO](coco_oc.json)

[OC@VCR](vcr_oc.json)

The annotation are formatted as: 
```
{
    "image_id": 
    "text_in":          # Question
    "text_out":         # Ground-Truth Answer
    "n_obj_exist":      # The total number of objects appeared in the image
}
```

## Multi-Class Identification
This task asks the model to identify if a certain object appeared in the image
We construct this task based on MS-COCO and Visual Commonsense Reasoning (VCR) datasets.

The annotation files are

[MCI@COCO](coco_mci.json)

[MCI@VCR](vcr_mci.json)

The annotation are formatted as: 
```
{
    "image_id": 
    "text_in":          # Question
    "text_out":         # Ground-Truth Answer
    "n_obj_exist":      # The total number of objects appeared in the image
}
```

