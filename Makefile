clean:
	rm -rf selected_bbox/*

requirement:
	pipreqs --ignore spot-sdk,TextCondRobotFetch --force --mode no-pin .

.PHONY: clean requirement