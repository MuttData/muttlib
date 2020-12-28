git fetch origin test_changelog
git fetch origin master
added_lines=$(git diff --numstat origin/master origin/test_changelog -- CHANGELOG.md | awk '{print $1}')
if [ -z $added_lines ] || [ $added_lines -eq 0 ]; then echo "Changelog has not been modified" && exit 1; else echo "Changelog has been modified" && exit 0; fi;
