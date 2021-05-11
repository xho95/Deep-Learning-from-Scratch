from LayerNaive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

# Layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# Forwards
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print("Multiply Layer / Forward ---------------")
print("price: ", price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("Multiply Layer / Backward ---------------")
print("dapple: ", dapple, ", dapple_num: ", dapple_num, ", dtax: ", dtax)
