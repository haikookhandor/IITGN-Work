clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    __int128 nanoseconds = end.tv_nsec - start.tv_nsec;
    cout << "\nTime taken: " << nanoseconds << " nanoseconds\n";