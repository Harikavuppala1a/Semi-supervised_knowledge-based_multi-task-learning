input_file = "data/slurs_source_text.txt"
output_file = "data/keywords_slurs.txt"
with open(input_file, "r") as f:
	content = f.read().replace('\n', '')

# def filter_out(cand):
# 	if '(' in cand or ')' in cand or '<span lang=' in cand:
# 		return True
# 	return False

cands = content.split('<tr>')
with open(output_file, "w") as f:
	for cand in cands:
		count = cand.count("<td")
		if count < 5 or cand.startswith('<td') == False:
			continue
		ind = cand.find('</td>')
		cand = cand[:ind]
		# if filter_out(cand):
		# 	continue
		parts = cand.split('<')
		# print(parts)
		t = parts[0]
		for p in parts[1:]:
			ind_2 = p.find('>')
			t+=p[ind_2+1:]
			# ind_3 = p[ind_2+1:].find('</a>')
			# ind_23 = ind_2+1+ind_3
			# t+=p[ind_2+1:ind_23] + p[ind_23+4:]
		for k1 in t.split(','):
			for k2 in k1.split('/'):
				f.write("%s\n" % k2.strip())
