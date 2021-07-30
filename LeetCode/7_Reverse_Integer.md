## 题目描述
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
## 代码
### C++
```cpp
class Solution {
public:
    int reverse(int x) {
        long long re = 0;
        long long t = x;
        int flag = 1;
        if(t < 0){flag = -1;t=-t;}
        while(t){
            re = re * 10 + t%10;
            t = t / 10;
        }
        re = flag * re;
        if(re < INT_MIN || re > INT_MAX){
            return 0;
        }
        return re;
    }
};
```
