awk '{B[NR%3]=$0} NR>2{ print B[(NR+1)%3]} /^level/ {print ""}  END {print B[(NR+2)%3]; print B[(NR+3)%3]}' dnd_spell.txt > dnd_spell_split.txt
