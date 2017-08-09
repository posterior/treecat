
add_executable(treecat_train train.cpp)
target_link_libraries(train
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
)
add_test(NAME trainer COMMAND trainer_test)
