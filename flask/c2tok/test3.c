#include <stdio.h>

int main(void)
{
	char pre[] = "4213657";
	char in[] = "1234567";
//	char str[] = "helloworld";
	char str[] = "Don't ask what your country can do for you, but ask what you can do for your country.\n";
	int i = 0;
	int pos = 0;
	int state = 0;	// init state
	int input = 0;	// 0: space		1: alpha
	link root = NULL;
	printf("hello, btree!\n");
	printf("%s", str);
	while (1)
	{
		char c;
		char wordbuf[32];
		c = str[pos++];
		if (c == '\0')
			break;
		input = get_input_type(c);
		//printf("state = %d ", state);
		
#if 1
		if (state == 0 && input == 1)
		{
			state = 1;
			//printf("word begin with <%c>\n", c);
			i = 0;
			wordbuf[i++] = c;
		} else if ($$)
		{
			wordbuf[i] = '\0';
			printf("word end value = <%s>\n", wordbuf);
			printf("insert %s(%d)\n", wordbuf, input);
			root = insert(root, wordbuf);
			state = 0;
		} else if (state == 1 && input == 1)
			wordbuf[i++] = c;
#endif
	}
#if 0
	for (i = 0; i < strlen(str); i++)
	{
		printf("insert %c \n", str[i]);
		root = insert(root, str[i]);
	}
#endif
	//pre_order(root);
	//printf("\npre travel finished\n");
	in_order(root, 0, '^');
	printf("\nin travel finished\n");
	//post_order(root);
	//printf("\npost travel finished\n");
	return 0;
}
