echo 'teste'

for i in `seq 1 10`;
do
	echo "Folder" $i
	mkdir $i
done

#for((i=1; i<= 200; i++)); do
#	echo 'aaa'
#	mogrify -resize 50% *.jpg        
#done

