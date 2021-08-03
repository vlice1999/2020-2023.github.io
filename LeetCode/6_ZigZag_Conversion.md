## 题目描述
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)


P   A   H   N


A P L S I I G


Y   I   R


And then read line by line: "PAHNAPLSIIGYIR"


Write the code that will take a string and make this conversion given a number of rows:


string convert(string s, int numRows);
## 解题思路
单纯的进行数学分析就可以。首先分类讨论：


第0行和第(numRows-1)行，每一行的公式为 t := t + 2 * (numRows - 1);


其第j行的公式为 t := t + 2 * (numRows - j - 1), t := t - 2 * (numRows - j - 1) + 2 * (numRows - 1);
## 代码
### C++(4ms)
```cpp
class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows == 1)
            return s;
        string re="";
        int len = s.size();
        int j = 1;
        for(int i = 0;i < numRows;i ++){
            int t = i;
            if(i == 0 || i == numRows-1){
                while(t < len){
                    re += s[t];
                    t = t + 2 * (numRows - 1);
                }
            }
            else{
                while(t < len){
                    re += s[t];
                    t = t + 2 * (numRows - j-1);
                    if(t >= len)
                        break;
                    re += s[t];
                    t = t - 2 * (numRows - j-1) + 2 * (numRows - 1);
                }
                j ++;
            }
        }
        return re;
    }
};
```
