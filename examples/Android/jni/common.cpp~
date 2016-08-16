#include "common.h"
//==========================================================================
// 2016/05/10
//==========================================================================

void debug(const char *s,...)
{
    va_list va; va_start(va,s);
    char buffer[debug_message_max];
	vsprintf(buffer,s,va); 
    va_end(va); 
	DEBUG_MACRO(buffer);
}



void cmdToArgv(std::string cmd , std::vector<char*> &v){
    std::istringstream ss(cmd);
    std::string arg;
    std::list<std::string> ls;
    while (ss >> arg)
    {
       //debug("arg = %s", arg.c_str());
       ls.push_back(arg);

       int c_size = sizeof(char)*ls.back().length();
       //debug("c_size = %d",c_size);
       char *c = (char*)malloc(c_size+1); // copy the null char in the end also
       memcpy(c, const_cast<char*>(ls.back().c_str()), c_size+1);
       

       //v.push_back(const_cast<char*>(ls.back().c_str()));
       v.push_back(c);
    }
    //v.push_back(0);  // need terminating null pointer
}
