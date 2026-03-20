#ifndef PARSER_H
#define PARSER_H
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
class Parser
{
private:
    std::string _filename;
public:
    
    Parser(std::string filename);
    ~Parser();
    std::vector<std::vector<double>> read_data();
};

#endif // PARSER_H