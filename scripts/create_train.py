# Python 3.7.6

import csv
import geopy.distance 

all_rows = [] 
with open("language.csv") as f:
         d2 = csv.DictReader(f) 
         for r in d2: 
             all_rows.append(r) 

# select those languages having more than 3 features
all4 = []
for l in all_rows:
    new_row = {}
    f = list(l.items())[10:]
    c = 0
    ff = []
    for t  in f:
        if t[1] != "":
            c += 1
            ff.append(t)
    if c > 3:
          m = list(l.items())[0:10]
          for b in m:
              new_row[b[0]] = b[1]
          for h in ff:
              new_row[h[0]] = h[1]
          all4.append(new_row)

# find geo-coordinates of Mayan languages
mayan_coo = []
for m in all4:
    if m["family"] == "Mayan":
      cc = (m["latitude"], m["longitude"])
      mayan_coo.append(cc)


# find the languages not being Mayan and being distant more than 1000 km
pre_fin = [] 
for m in all4: 
      coord1 = (m["latitude"], m["longitude"]) 
      for coord2 in mayan_coo: 
         dist = geopy.distance.geodesic(coord1, coord2) 
      if m["family"] != "Mayan" and  dist > 1000: 
         w = m['wals_code'] 
         fa = m['family']  
         f = list(m.items())[10:] 
         pre_fin.append([w, fa, f]) 

#  find the set of all features of the above languages
all_feat = set()
for a,b,c in pre_fin: 
    for k in c: 
         all_feat.add(k[0]) 

# extract from all_feat above the features that are in more than 9 languages
fts = {}
for f in all_feat:
   count = 0
   for a,b,c in pre_fin:
      for d,e in c:
          if f == d:
              count += 1
   if count > 9:
     fts[f] = count

# create the list of languages not being Mayan, being distant more than 1000 km, with more than 3 features,  
# with features being in more than 9 languages
a = list(fts.keys())
fin = []
for m in all4:
    coord1 = (m["latitude"], m["longitude"])
    for coord2 in mayan_coo:
       dist = geopy.distance.geodesic(coord1, coord2)
    if m["family"] != "Mayan" and  dist > 1000:
       w = m['wals_code']
       name = m["Name"]
       lat = m["latitude"]
       lon = m["longitude"]
       gen = m["genus"]
       fa = m['family'] 
       ccod = m["countrycodes"]
       f = list(m.items())[10:]
       nn = []
       for n in f:
           if n[0] in a:
             j = n[0].split(" ", 1)
             s = j[1].replace(" ", "_") + "=" + n[1]
             nn.append(s)
       if len(nn) > 3:  # to be sure the features are at least 4
         vv = "|".join(nn)
         fin.append(w + "\t" + name + "\t" + lat + "\t" + lon + "\t" + gen + "\t" + fa + "\t" + ccod + "\t" + vv)

# create the file
with open("train.csv", "wt") as f:
    print("wals_code	name	latitude	longitude	genus	family	countrycodes	features", file=f)
    for g in fin:
        print(g, file=f)
