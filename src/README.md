# Config file (VERY IMPORTANT)
The json file MUST resemble a dict in this form:
```
{
    "face": [],
    "mouth": [],
    "l_hand": [],
    "r_hand": [],
    "body": []
}
```
where the lists are lists of points that you want to inclue in traininig

**FIRST POINT IN THE LIST IS THE ROOT POINT ACCORDING TO WICH THE NORMALIZATION SHOULD TAKE PLACE**
**BOTH l_hand AND r_hand NEED TO BE THE SAME**