Should use time variant so the attack is not as obvious at beginning

9/4
- Create error variable for nominal and obs to make sure it is 0. Internal dynamics should not affect the error if perfect.
    - This could be a monitoring function (this is used in many other papers)
- Apply smsf
    - An attack would change the orientation of the drone in a nonlinear way. This cannot be easily observed in the linear dynamics
    - attack scaling factor should be same between drones
- Test different types of attacks with avoidance control and see what happens. Do any violate the perfectly undetectable-ness?