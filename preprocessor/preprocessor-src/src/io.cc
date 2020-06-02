#include "io.hh"

namespace __io__ {
void close(std::ifstream& file) {
    if(file.is_open())
    	file.close();
}
void close(std::ofstream& file) {
    if(file.is_open())
    	file.close();
}
void close(std::fstream& file) {
    if(file.is_open())
    	file.close();
}

std::vector<std::string> read(const std::string& file_name) {
    std::ifstream file=open(file_name);
    std::vector<std::string> lines;
    if(file.is_open()) {
        std::string line;
        while(std::getline(file,line))
            lines.push_back(line);
        close(file);
    }
    return lines;
}
void write(const std::vector<std::string>& lines,const std::string& file_name) {
    auto file=open<1>(file_name);
    for( const std::string& line : lines )
    	file<<line<<std::endl;
}

bool create_directory(const std::string &file_name) {
	return std::filesystem::create_directories(std::filesystem::path(file_name));
}
bool remove_file(const std::string& file_name) {
	std::filesystem::path path(file_name);
	return std::filesystem::remove(path);
}
bool remove_directory(const std::string &file_name) {
	std::filesystem::path path(std::filesystem::absolute(std::filesystem::path(file_name)));
	if(path!=std::filesystem::path("/"))
		return (std::filesystem::remove_all(path)>0);
	return false;
}

std::vector<std::string> toLines(const std::string& string) {
	std::istringstream instring(string);
	std::string line;
	std::vector<std::string> lines;
	while(std::getline(instring,line))
		if(!line.empty())
			lines.push_back(line);
	return lines;
}
}

