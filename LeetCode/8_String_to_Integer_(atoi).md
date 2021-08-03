## 题目描述
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

The algorithm for myAtoi(string s) is as follows:

Read in and ignore any leading whitespace.
Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
Read in next the characters until the next non-digit charcter or the end of the input is reached. The rest of the string is ignored.
Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -231 should be clamped to -231, and integers greater than 231 - 1 should be clamped to 231 - 1.
Return the integer as the final result.
## 解题思路
略..
## 代码
### C++(0ms)
```cpp
class Solution {
public:
    int myAtoi(string s) {
        int t;
        bool flag = false;
        int c = 1;
        long long re = 0;
        int o = '0';
        for(int i = 0;i < s.size();i ++){
            t = s[i];
            if(flag == false){
                if(s[i] != '-' && s[i] != '+' && s [i] != ' ' &&  (s[i] < '0' || s[i] > '9')){
                    return 0;
                }
            }
        
            if(s[i] == ' ' && flag == false){
                continue;
            }            
            if(flag == true && (s[i] < '0' || s[i] > '9')){
                if(c*re < INT_MIN)
                    return INT_MIN;
                if(c*re > INT_MAX)
                    return INT_MAX;
                return c * re;
            }
            if(s[i] == '-' && flag == false){
                flag = true;
                c = -1;
            }
            if((s[i] == '0' || s[i] == '+')&& flag == false){
                flag = true;
            }
            
            if(s[i] >= '0' && s[i] <= '9'){
                flag = true;
                if(re > INT_MAX )
                    break;
                re = re * 10 + (s[i] - o);
            }
            if(s[i] == '.'){
                break;
            }
        }
        
        if(c*re < INT_MIN)
            return INT_MIN;
        if(c*re > INT_MAX)
            return INT_MAX;
        return c * re;
    }
};
```
### C++(4ms)
```cpp
class Solution {
public:
    int myAtoi(string s) {
        long long re = 0;
        int index = 0;
        int flag = 1;
        while(s[index] == ' '){
            index ++;
        }
        if(s[index] == '-'){
            flag = -1;
            index ++;
        }
        else if(s[index] == '+'){
            flag = 1;
            index ++;
        }
        while(index < s.size()){
            if (s[index] < '0' or s[index] > '9'){
                return flag*re;
            }
            else{
                long long temp = (re * 10 + (s[index] - '0'));
                if(flag * temp > INT_MAX){
                    return INT_MAX;
                }
                if(flag * temp < INT_MIN){
                    return INT_MIN;
                }
                re = temp;
                index ++;
            }
        }
        if(flag * re > INT_MAX){
            return INT_MAX;
        }
        if(flag * re < INT_MIN){
            return INT_MIN;
        }
        return flag*re;
    }
};
      
