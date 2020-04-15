#include "utils.h"

void getFiles(const string folder, vector<string>& result) {
	list<string> subfolders;
	subfolders.push_back(folder);
#ifdef WIN32
	while (!subfolders.empty()) {
		//subfolders.back()：返回最后一个元素
		std::string current_folder(subfolders.back());

		if (*(current_folder.end() - 1) != '/') {
			current_folder.append("/*");
		}
		else {
			current_folder.append("*");
		}

		//删除最后一个元素
		subfolders.pop_back();
		//用于存储各种文件信息
		struct _finddata_t file_info;
		//返回值： 
		//如果查找成功的话，将返回一个long型的唯一的查找用的句柄（就是一个唯一编号）。这个句柄将在_findnext函数中被使用。若失败，则返回-1。 
		//参数： 
		//filespec：标明文件的字符串，可支持通配符。比如：*.c，则表示当前文件夹下的所有后缀为C的文件。 
		//fileinfo ：这里就是用来存放文件信息的结构体的指针。这个结构体必须在调用此函数前声明，不过不用初始化，只要分配了内存空间就可以了。 
		//函数成功后，函数会把找到的文件的信息放入这个结构体中。
		//调用此函数后file_info保存的是指定路径下的第一个文件，如a1.png
		//https://blog.csdn.net/wangqingchuan92/article/details/77979669
		auto file_handler = _findfirst(current_folder.c_str(), &file_info);
		//strcmp：字符串比较函数
		//该函数返回值如下：
		//如果返回值小于 0，则表示 str1 小于 str2。
		//如果返回值大于 0，则表示 str1 大于 str2。
		//如果返回值等于 0，则表示 str1 等于 str2。
		while (file_handler != -1) {
			//判断是否为"."当前目录，".."上一层目录
			if ((!strcmp(file_info.name, ".") || !strcmp(file_info.name, ".."))) {
				//int _findnext( long handle, struct _finddata_t *fileinfo );
				//返回值：若成功返回0，否则返回-1。
				//参数：
				//handle：即由_findfirst函数返回回来的句柄。
				//fileinfo：文件信息结构体的指针。找到文件后，函数将该文件信息放入此结构体中。
				//调用此函数后file_info保存的是指定路径下的下一个文件，如a2.png
				if (_findnext(file_handler, &file_info) != 0) break;
				continue;
			}
			//判断是否有子目录 
			if (file_info.attrib & _A_SUBDIR) {

				std::string folder(current_folder);
				folder.pop_back();
				folder.append(file_info.name);

				subfolders.push_back(folder.c_str());

			}
			else {
				std::string file_path;
				// 赋以新值 
				file_path.assign(current_folder.c_str());
				//1.3 std::string::pop_back
				//原型：void pop_back();
				//说明：删除源字符串的最后一个字符，有效的减少它的长度。
				file_path.pop_back();
				file_path.append(file_info.name);

				result.push_back(file_path);

			}
			//寻找下一个文件
			//调用此函数后file_info保存的是指定路径下的下一个文件，如a2.png
			if (_findnext(file_handler, &file_info) != 0) break;
		}
		//int _findclose( long handle );
		// 返回值：成功返回0，失败返回-1。
		// 参数：
		//  handle ：_findfirst函数返回回来的句柄。
		//该结构体和搭配的函数使用的逻辑就是先用_findfirst查找第一个文件，若成功则用返回的句柄，
		//调用_findnext函数查找其他的文件，当查找完毕后用，用_findclose函数结束查找。
		_findclose(file_handler);
	}
#else
	while (!subfolders.empty()) {
		string current_folder(subfolders.back());
		if (*(current_folder.end() - 1) != '/') {
			current_folder.push_back('/');
		}
		DIR* pdir = opendir(current_folder.c_str());

		subfolders.pop_back();

		if (!pdir) {
			continue;
		}

		dirent* dir = NULL;

		while ((dir = readdir(pdir)) != NULL) {
			struct stat st;
			if (!strcmp(dir->d_name, ".") || !strcmp(dir->d_name, "..")) {
				continue;
			}
			if (!strcmp(dir->d_name, ".DS_Store")) {
				continue;
			}

			std::string file_path;

			file_path.append(current_folder.c_str());
			file_path.append(dir->d_name);

			if (lstat(file_path.c_str(), &st) < 0) {
				continue;
			}

			if (S_ISDIR(st.st_mode)) {
				std::string subfolder(current_folder);
				subfolder.append(dir->d_name);
				subfolders.push_back(subfolder.c_str());
			}
			else {
				result.push_back(file_path);


			}
		}
		closedir(pdir);
	}
#endif
}