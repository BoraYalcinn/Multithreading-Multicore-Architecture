#include <curl/curl.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>

size_t writeCall(void* contents, size_t size, size_t nmemb, std::string* output){
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

void downloadURL(const char* url) {
    CURL* curl = curl_easy_init();
    std::string result;

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCall);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl error: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
    }
}

int main(){

    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::vector<std::string> URLs;
    for (int i = 1; i <= 12; i++) {
        URLs.push_back("http://localhost:8001/file" + std::to_string(i) + ".txt");
    }

    std::vector<int> poolSizes = {1, 2, 4, 6, 12};
    std::cout << "----------PROCESS-----------" << std::endl ;
    for(int poolSize: poolSizes){

        int urlsPerProcess = URLs.size() / poolSize;
        int remainder = URLs.size() % poolSize;

        std::vector<pid_t> children;
        int currentIndex = 0;

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < poolSize; i++) {
            int urlsForThisProcess = 0;
            if (remainder != 0) {
                urlsForThisProcess = urlsPerProcess + 1;
                remainder--;
            }else {
                urlsForThisProcess = urlsPerProcess;
            }

            int startIdx = currentIndex;
            int endIdx = currentIndex + urlsForThisProcess;
            currentIndex = endIdx;

            pid_t pid = fork();

            if (pid == 0) {
                // CHILD process
                curl_global_init(CURL_GLOBAL_DEFAULT);
                for (int j = startIdx; j < endIdx; j++) {
                    downloadURL(URLs[j].c_str());
                }
                curl_global_cleanup();
                _exit(0);
            } else {
                children.push_back(pid);
            }
        }

        for (const auto& child : children) {
            waitpid(child, nullptr, 0);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> timeElapsed = end - start;

        std::cout << "Pool size = " << poolSize << " -> Time elapsed = " << timeElapsed.count() << "s" << std::endl;
        std::cout << "-----------------------------" << std::endl ;
        }

    curl_global_cleanup();

    return 0;
}