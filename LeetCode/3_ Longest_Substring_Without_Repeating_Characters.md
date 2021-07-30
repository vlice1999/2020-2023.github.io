## 题目描述
Given a string s, find the length of the longest substring without repeating characters.

 
#### Example 1:
Input: s = "abcabcbb"


Output: 3


Explanation: The answer is "abc", with the length of 3.

## 解题思路
使用C++ string类型本身的find函数，可以判断字符串中是否存在某个字母，如果存在的话可以返回其位置。可以设置一个temp字符串用于记录最长子串，然后用find函数判断是否存在重复字母，然后用string的
substr函数截取字符串，并更新最长不重复字符串长度。

## 代码
### C++
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        string temp = "";
        int re = 0;
        for(int i = 0;i < s.size();i ++){
            if(temp.find(s[i]) == string::npos){
                temp += s[i];
            }
            else{
                re = re > temp.size()?re:temp.size();
                temp = temp.substr(temp.find(s[i])+1)+s[i];
            }
        }
        return re>temp.size()?re:temp.size();
    }
};
```
