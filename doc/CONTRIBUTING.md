How to contribute
========

Thanks for taking the time to contribute to tiny-dnn! The following is a few guidelines for contributors.
These are just guidelines, not rules, and feel free to propose changes to this document in a pull request. 

## Getting Started
- Make sure you have a C++11 compiler.
- Make sure you have a GitHub account.
- Register a report about your issue.
    - Check [the issue list](https://github.com/tiny-dnn/tiny-dnn/issues) to see if the problem has already been reported.
    - This can be skipped if the issue is trivial (fixing a typo, etc).

## Making Changes
- Create a topic branch where you want to base your work.
    - This is usually the ```master``` branch.
- Make commits.
- Make sure you have added the necessary tests for your changes.
- Submit a pull request.
- Make sure all CI builds are passed.

## Coding guides
- Keep header-only
- Keep dependency-free
    - If your change requires 3rd party libraries, this should be __optional__ in tiny-dnn.
    Please guard your 3rd party dependent code by ```#ifdef - #endif``` block, and white CMakelist option to enable the block - 
    but lesser these switches, the better.
- Keep platform-independent
    - Use C++ standard library instead of Windows/POSIX dependent API
    - CPU/GPU optimized code should be extracted as a separated file, and should be guarded as preprocessor macro.

### Preferred coding style 
- Use ```snake_case``` for identifiers
- Use ```SCREAMING_SNAKE_CASE_START_WITH_CNN``` for preprocessor macros.
- Use 4-spaces instead of tabs.
