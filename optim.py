import numpy as np

class Adagrad:
    # γ::Float         # learning rate
    # ϵ::Float         # small value for numerical stability
    # g::Array{Float}  # sum of squares of past gradients
    # size::tuple of sizes
            
    def __init__(self, γ, size):
    	# size eg: (3,), (3,1), (3,2)...

        self.γ = γ
        self.ϵ = 1.0e-8
        self.g = np.zeros(size)
        
    def update(self, val, grad):
        self.g += grad**2

        #dd = - self.γ * grad / np.sqrt( self.g + self.ϵ )
        #print(dd[0:3])
        return val - self.γ * grad / np.sqrt( self.g + self.ϵ )



class Adadelta:
	# α::Float    			# influence parameter, eg 0.9
	# ϵ::Float    			# small value for numerical stability
	# g::Array{Float64}  	# decaying sum of squares of past (raw) gradients
	# f::Array{Float64}  	# decaying sum of squares of past (weighted) gradients

	def __init__(self, α, size, γ=1.0, ϵ = 1.0e-8):
		# size eg: (3,), (3,1), (3,2)...

		self.ϵ = ϵ 
		self.α = α
		self.g = np.zeros(size)
		self.f = np.zeros(size)
		self.γ = γ      # in the original formulation, this is 1

	def update(self, val, grad):

		α = self.α
		self.g = α * self.g + (1-α)*grad**2
		delta = - grad * np.sqrt(self.f + self.ϵ) / np.sqrt(self.g + self.ϵ)
		self.f = α * self.f + (1-α)*delta**2

		#print(np.sqrt(self.f + self.ϵ) / np.sqrt(self.g + self.ϵ))

		#print(delta[0:3])
		return val + delta*self.γ



class RMSprop:
	# γ::Float 				# learning rate
	# α::Float    			# influence parameter of past, eg 0.9
	# ϵ::Float    			# small value for numerical stability
	# g::Array{Float64}  	# decaying sum of squares of past (raw) gradients

	def __init__(self, size, γ = 0.001, α=0.9, ϵ = 1.0e-8):
		# size eg: (3,), (3,1), (3,2)...

		self.ϵ = ϵ 
		self.α = α
		self.γ = γ
		self.g = np.zeros(size)

	def update(self, val, grad):

		self.g = self.α * self.g + (1-self.α)*grad**2
		delta = - self.γ * grad / np.sqrt(self.g + self.ϵ)

		#print(delta[0:3])
		return val + delta




class Adam:
	# γ::Float 				# learning rate
	# β1::Float    			# influence parameter of the past for the first moment
	# β2::Float    			# influence parameter of the past for the second moment
	# ϵ::Float    			# small value for numerical stability
	# m::Array{Float64}  	# past first moment
	# v::Array{Float64}  	# past second moment

	def __init__(self, γ, size, β1 = 0.9, β2 = 0.999, ϵ = 1.0e-8):
		# size eg: (3,), (3,1), (3,2)...

		self.γ = γ
		self.ϵ = ϵ 
		self.β1 = β1
		self.β2 = β2
		self.m = np.zeros(size)
		self.v = np.zeros(size)
		self.t = 1

	def update(self, val, grad):


		self.m = self.β1 * self.m + (1-self.β1) * grad
		self.v = self.β2 * self.v + (1-self.β2) * grad**2

		mh = self.m / (1 - self.β1**self.t)
		vh = self.v / (1 - self.β2**self.t)

		delta = - self.γ * mh / (np.sqrt(vh) + self.ϵ)

		self.t += 1

		return val + delta




class HeurSign:
	# γ::Float 						# initial learning rate
	# γs::Array{Float64}  			# current individual learning rates
	# pastSign::Array{Float64}  	# past signs of gradients

	def __init__(self, size, γ, γInc = 1.02, γDec = 0.5):
		# size eg: (3,), (3,1), (3,2)...

		self.γs = np.ones(size)*γ
		self.pastSign = np.zeros(size)
		self.γInc = γInc
		self.γDec = γDec

	def update(self, val, grad):


		delta = - self.γs * grad

		signs = grad>0.0
		eqSign = signs == self.pastSign
		mults = eqSign * self.γInc + (1-eqSign)*self.γDec
		self.γs = self.γs*mults
		self.pastSign = signs

		return val + delta
